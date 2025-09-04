#ifndef KERNEL_CUH
#define KERNEL_CUH

__global__ void data_reset_kernel(const float4 *d_in, float4 *d_out, const int n, const int m);

__global__ void stream_and_persistent_kernel(const float4* __restrict__ d_in_persistent, const float4* __restrict__ d_in_streaming, float4* __restrict__ d_out, const int n, const int m);

__global__ void stream_and_persistent_kernel_batched(const float4* __restrict__ d_in_persistent, const float4* __restrict__ d_in_streaming, float4* __restrict__ d_out, const int n, const int m);
#endif // KERNEL_CUH