#include <cstdlib>
#include <iostream>
#include <vector>

#include "copy.cuh"   
#include "cuda_utils.cuh"    
#include "verify_result.hpp"

int main(int argc, char* argv[]) {
    int N = 1 << 30;
    int device_id = 0;
    if (argc > 1) device_id = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);

    const std::vector<float> in(N, 5);
    std::vector<float> out(N);
    
    run_copy(in.data(), out.data(), N, device_id);

    verify_copy(in, out);

    return 0;
}
