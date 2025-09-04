#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "data_resetting.cuh"   
#include "cuda_utils.cuh"    
#include "verify_result.hpp"

int main(int argc, char* argv[]) {
    int N = 1 << 30; // 4096MB
    int M = 1 << 21; // 2^22 -> 16MB
    int device_id = 0;
    if (argc > 1) device_id = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) M = std::atoi(argv[3]);

    std::vector<float> persistent_data(M);
    std::vector<float> streaming_data(N);
    std::vector<float> output_data(N);

    //std::mt19937 eng {42};  
    std::minstd_rand eng {42};
    std::normal_distribution<float> dist {0, 10};
    auto gen = [&](){ return dist(eng); };
    std::generate(persistent_data.begin(), persistent_data.end(), gen);
    std::generate(streaming_data.begin(), streaming_data.end(), gen);

    run_stream_and_persistent_batched(persistent_data.data(), streaming_data.data(), output_data.data(), N, M, device_id);

    verify_stream_and_persistent(persistent_data, streaming_data, output_data);

    return 0;
}
