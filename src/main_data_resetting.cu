#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "data_resetting.cuh"   
#include "cuda_utils.cuh"    
#include "verify_result.hpp"

int main(int argc, char* argv[]) {
    int N = 1 << 30; // 8192MB
    int M = 1 << 22; // 2^22 -> 16MB
    int device_id = 0;
    if (argc > 1) device_id = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) M = std::atoi(argv[3]);

    std::vector<float> in(M);
    std::vector<float> out(N);

    std::mt19937 eng {42};  
    std::normal_distribution<float> dist {0, 10};
    auto gen = [&](){ return dist(eng); };
    std::generate(in.begin(), in.end(), gen);

    run_data_reset(in.data(), out.data(), N, M, device_id);

    verify_data_reset(in, out);

    return 0;
}
