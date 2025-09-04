cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(L2Persistent LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90a)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_FLAGS "--generate-line-info --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -arch=sm_90a")
set(CMAKE_CUDA_FLAGS "--generate-line-info --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing")
set(CMAKE_CXX_FLAGS "-O3")

add_library(gpu_lib SHARED
    gpu/cuda_utils.cu
    gpu/copy.cu
    gpu/data_resetting.cu
    gpu/kernel.cu
)
target_include_directories(gpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gpu)

add_library(utils SHARED 
    utils/verify_result.cpp
)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils)

add_executable(main_copy src/main_copy.cu)
target_link_libraries(main_copy PRIVATE gpu_lib utils)

add_executable(main_data_resetting src/main_data_resetting.cu)
target_link_libraries(main_data_resetting PRIVATE gpu_lib utils)

add_executable(main_data_resetting_persistence src/main_data_resetting_persistence.cu)
target_link_libraries(main_data_resetting_persistence PRIVATE gpu_lib utils)

add_executable(main_stream_and_persistent src/main_stream_and_persistent.cu)
target_link_libraries(main_stream_and_persistent PRIVATE gpu_lib utils)

add_executable(main_stream src/main_stream.cu)
target_link_libraries(main_stream PRIVATE gpu_lib utils)

add_executable(main_stream_and_persistent_batched src/main_stream_and_persistent_batched.cu)
target_link_libraries(main_stream_and_persistent_batched PRIVATE gpu_lib utils)
