#include <iomanip>
#include <iostream>

#include "cuda_utils.cuh"
#include "data_resetting.cuh"
#include "kernel.cuh"

void run_data_reset(const float*  in, float* out, const int n, const int m, const int device_id) {
    const size_t size_in = m * sizeof(float);
    const size_t size_out = n * sizeof(float);
    float *d_in, *d_out;
    cudaStream_t stream;
    cudaDeviceProp prop;

    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Running on device = " << device_id << std::endl;
    std::cout << "In data size = " << (size_in) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Out data size = " << (size_out) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "L2 cache size = " << prop.l2CacheSize / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Maximum persistent L2 cache size = " << prop.persistingL2CacheMaxSize / (1024.0 * 1024.0) << "MB" << std::endl;
    CHECK_CUDA_ERROR(cudaSetDevice(device_id));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in, size_in));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, size_out));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, in, size_in, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    const int threads_per_block = 1024;
    const int blocks_per_grid = (n/4 + threads_per_block - 1) / threads_per_block;
    data_reset_kernel<<<blocks_per_grid, threads_per_block, 0U, stream>>>(
        reinterpret_cast<float4 *>(d_in), reinterpret_cast<float4 *>(d_out), n, m
    );

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost));
  
    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}


void run_data_reset_with_l2_persistence(const float*  in, float* out, const int n, const int m, const int device_id) {
    const size_t size_in = m * sizeof(float);
    const size_t size_out = n * sizeof(float);
    float *d_in, *d_out;
    cudaStream_t stream;
    cudaDeviceProp prop;

    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Running on device = " << device_id << std::endl;
    std::cout << "In data size = " << (size_in) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Out data size = " << (size_out) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "L2 cache size = " << prop.l2CacheSize / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Maximum persistent L2 cache size = " << prop.persistingL2CacheMaxSize / (1024.0 * 1024.0) << "MB" << std::endl;
    CHECK_CUDA_ERROR(cudaSetDevice(device_id));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in, size_in));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, size_out));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, in, size_in, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    size_t l2_max_cache_persistent = 8 * 1024 * 1024;//1 * 1024 * 1024; 
    std::cout << "Set Limit for persisting L2 cache size equal to " << l2_max_cache_persistent / (1024 * 1024) << "MB" << std::endl;
    CHECK_CUDA_ERROR(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_max_cache_persistent));

    std::cout << "Set stream attributes for persisting L2 cache." << std::endl;
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(d_in);
    stream_attribute.accessPolicyWindow.num_bytes = l2_max_cache_persistent;
    //stream_attribute.accessPolicyWindow.hitRatio  = 1;
    stream_attribute.accessPolicyWindow.hitRatio = std::min(l2_max_cache_persistent / ((float) size_in), 1.0f);
    std::cout << "Hit Ratio = " << stream_attribute.accessPolicyWindow.hitRatio << std::endl;
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

    CHECK_CUDA_ERROR(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

    const int threads_per_block = 1024;
    const int blocks_per_grid = (n/4 + threads_per_block - 1) / threads_per_block;
    data_reset_kernel<<<blocks_per_grid, threads_per_block, 0U, stream>>>(
        reinterpret_cast<float4 *>(d_in), reinterpret_cast<float4 *>(d_out), n, m
    );

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost));
  
    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

void run_stream_and_persistent(const float*  in_persistent, const float*  in_streaming, float* out, const int n, const int m, const int device_id) {
    const size_t size_in = m * sizeof(float);
    const size_t size_out = n * sizeof(float);
    float *d_in_persistent, *d_in_streaming, *d_out;
    cudaStream_t stream;
    cudaDeviceProp prop;

    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Running on device = " << device_id << std::endl;
    std::cout << "Persistent data size = " << (size_in) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Streaming data size = " << (size_out) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Out data size = " << (size_out) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "L2 cache size = " << prop.l2CacheSize / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Maximum persistent L2 cache size = " << prop.persistingL2CacheMaxSize / (1024.0 * 1024.0) << "MB" << std::endl;
    CHECK_CUDA_ERROR(cudaSetDevice(device_id));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in_persistent, size_in));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in_streaming, size_out));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, size_out));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in_persistent, in_persistent, size_in, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in_streaming, in_streaming, size_out, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    const int threads_per_block = 256;
    const int blocks_per_grid = (n/4 + threads_per_block - 1) / threads_per_block;
    stream_and_persistent_kernel<<<blocks_per_grid, threads_per_block, 0U, stream>>>(
        reinterpret_cast<float4 *>(d_in_persistent), reinterpret_cast<float4 *>(d_in_streaming), reinterpret_cast<float4 *>(d_out), n, m
    );

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost));
  
    CHECK_CUDA_ERROR(cudaFree(d_in_persistent));
    CHECK_CUDA_ERROR(cudaFree(d_in_streaming));
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

void run_stream_and_persistent_with_l2_persistence(const float*  in_persistent, const float*  in_streaming, float* out, const int n, const int m, const int device_id) {
    const size_t size_in = m * sizeof(float);
    const size_t size_out = n * sizeof(float);
    float *d_in_persistent, *d_in_streaming, *d_out;
    cudaStream_t stream;
    cudaDeviceProp prop;

    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Running on device = " << device_id << std::endl;
    std::cout << "Persistent data size = " << (size_in) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Streaming data size = " << (size_out) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Out data size = " << (size_out) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "L2 cache size = " << prop.l2CacheSize / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Maximum persistent L2 cache size = " << prop.persistingL2CacheMaxSize / (1024.0 * 1024.0) << "MB" << std::endl;
    CHECK_CUDA_ERROR(cudaSetDevice(device_id));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in_persistent, size_in));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in_streaming, size_out));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, size_out));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in_persistent, in_persistent, size_in, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in_streaming, in_streaming, size_out, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    size_t l2_max_cache_persistent = 8 * 1024 * 1024;//1 * 1024 * 1024; 
    std::cout << "Set Limit for persisting L2 cache size equal to " << l2_max_cache_persistent / (1024 * 1024) << "MB" << std::endl;
    CHECK_CUDA_ERROR(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_max_cache_persistent));

    std::cout << "Set stream attributes for persisting L2 cache." << std::endl;
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(d_in_persistent);
    stream_attribute.accessPolicyWindow.num_bytes = l2_max_cache_persistent;
    //stream_attribute.accessPolicyWindow.hitRatio  = 1;
    stream_attribute.accessPolicyWindow.hitRatio = std::min(l2_max_cache_persistent / ((float) size_in), 1.0f);
    std::cout << "Hit Ratio = " << stream_attribute.accessPolicyWindow.hitRatio << std::endl;
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

    CHECK_CUDA_ERROR(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

    const int threads_per_block = 256;
    const int blocks_per_grid = (n/4 + threads_per_block - 1) / threads_per_block;
    stream_and_persistent_kernel<<<blocks_per_grid, threads_per_block, 0U, stream>>>(
        reinterpret_cast<float4 *>(d_in_persistent), reinterpret_cast<float4 *>(d_in_streaming), reinterpret_cast<float4 *>(d_out), n, m
    );

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost));
  
    CHECK_CUDA_ERROR(cudaFree(d_in_persistent));
    CHECK_CUDA_ERROR(cudaFree(d_in_streaming));
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

void run_stream_and_persistent_batched(const float*  in_persistent, const float*  in_streaming, float* out, const int n, const int m, const int device_id) {
    const size_t size_in = m * sizeof(float);
    const size_t size_out = n * sizeof(float);
    float *d_in_persistent, *d_in_streaming, *d_out;
    cudaStream_t stream;
    cudaDeviceProp prop;

    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Running on device = " << device_id << std::endl;
    std::cout << "Persistent data size = " << (size_in) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Streaming data size = " << (size_out) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Out data size = " << (size_out) / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "L2 cache size = " << prop.l2CacheSize / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "Maximum persistent L2 cache size = " << prop.persistingL2CacheMaxSize / (1024.0 * 1024.0) << "MB" << std::endl;
    CHECK_CUDA_ERROR(cudaSetDevice(device_id));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in_persistent, size_in));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in_streaming, size_out));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, size_out));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in_persistent, in_persistent, size_in, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in_streaming, in_streaming, size_out, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    size_t l2_max_cache_persistent = 8 * 1024 * 1024;//1 * 1024 * 1024; 
    std::cout << "Set Limit for persisting L2 cache size equal to " << l2_max_cache_persistent / (1024 * 1024) << "MB" << std::endl;
    CHECK_CUDA_ERROR(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_max_cache_persistent));

    std::cout << "Set stream attributes for persisting L2 cache." << std::endl;
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(d_in_persistent);
    stream_attribute.accessPolicyWindow.num_bytes = l2_max_cache_persistent;
    //stream_attribute.accessPolicyWindow.hitRatio  = 1;
    stream_attribute.accessPolicyWindow.hitRatio = std::min(l2_max_cache_persistent / ((float) size_in), 1.0f);
    std::cout << "Hit Ratio = " << stream_attribute.accessPolicyWindow.hitRatio << std::endl;
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

    CHECK_CUDA_ERROR(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

    const int threads_per_block = 256;
    const int blocks_per_grid = (n/(4 * 2) + threads_per_block - 1) / threads_per_block;
    stream_and_persistent_kernel_batched<<<blocks_per_grid, threads_per_block, 0U, stream>>>(
        reinterpret_cast<float4 *>(d_in_persistent), reinterpret_cast<float4 *>(d_in_streaming), reinterpret_cast<float4 *>(d_out), n, m
    );

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost));
  
    CHECK_CUDA_ERROR(cudaFree(d_in_persistent));
    CHECK_CUDA_ERROR(cudaFree(d_in_streaming));
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
