#include <iostream>

#include "copy.cuh"
#include "cuda_utils.cuh"

__global__ void copy_kernel(const float4 *d_in, float4 *d_out, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n/4) d_out[i] = d_in[i];
}

void run_copy(const float*  in, float* out, const int n, const int device_id) {
    const size_t size = n * sizeof(float);
    float *d_in, *d_out;
    cudaStream_t stream;
    cudaDeviceProp prop;

    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "Running with N = " << n << " on device = " << device_id << std::endl;
    CHECK_CUDA_ERROR(cudaSetDevice(device_id));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    const int threads_per_block = 1024;
    const int blocks_per_grid = (n / 4 + threads_per_block - 1) / threads_per_block;
    copy_kernel<<<blocks_per_grid, threads_per_block, 0U, stream>>>(reinterpret_cast<float4*>(d_in), reinterpret_cast<float4*>(d_out), n);

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));
  
    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
