#include "kernel.cuh"

__global__ void data_reset_kernel(const float4 *d_in, float4 *d_out, const int n, const int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n/4) d_out[i] = d_in[i % (m/4)];
}


__global__ void stream_and_persistent_kernel(const float4* __restrict__ d_in_persistent, const float4* __restrict__ d_in_streaming, float4* __restrict__ d_out, const int n, const int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Note n / 4 == n >> 2 and m / 4 == m >> 2, i % m == i & (m-1) if m 2^x
    if (i < (n >> 2)) {
        //float4 s = __ldlu(&d_in_streaming[i]);
        float4 s = d_in_streaming[i];
        float4 p = __ldcg(&d_in_persistent[i & ((m >> 2)-1)]);
        d_out[i] = make_float4(p.x+s.x, p.y+s.y, p.z+s.z, p.w+s.w);
        //atomicAdd(&d_out[i], s);
        //atomicAdd(&d_out[i], p);
    }
}

__global__ void stream_and_persistent_kernel_batched(const float4* __restrict__ d_in_persistent, const float4* __restrict__ d_in_streaming, float4* __restrict__ d_out, const int n, const int m) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

#pragma unroll
    for(auto j = i; j < i + 2; j++) {
        if (j < n/4) {
            const float4 s = d_in_streaming[j];
            const float4 p = d_in_persistent[j % (m/4)];
            d_out[j] = make_float4(p.x+s.x, p.y+s.y, p.z+s.z, p.w+s.w);
        }
    }
}