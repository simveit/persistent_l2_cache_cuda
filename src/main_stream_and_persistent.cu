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
    int M = 1 << 23; // 2^21 -> 8MB
    int device_id = 0;
    if (argc > 1) device_id = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) M = std::atoi(argv[3]);

    std::vector<float> in_persistent(M);
    std::vector<float> in_stream(N);
    std::vector<float> out(N);

    //std::mt19937 eng {42};  
    std::minstd_rand eng {42};
    std::normal_distribution<float> dist {0, 10};
    auto gen = [&](){ return dist(eng); };
    std::generate(in_persistent.begin(), in_persistent.end(), gen);
    std::generate(in_stream.begin(), in_stream.end(), gen);

    run_stream_and_persistent(in_persistent.data(), in_stream.data(), out.data(), N, M, device_id);

    verify_stream_and_persistent(in_persistent, in_stream, out);

    return 0;
}
