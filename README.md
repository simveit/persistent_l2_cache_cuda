## Persistent L2 Cache

Blogpost:
[https://veitner.bearblog.dev/l2-benchmarking/
](https://veitner.bearblog.dev/gpu-l2-cache-persistence/)

This is a project to learn mainly about the L2 Cache and how we can set aside parts of it for persistence access using `CUDA`.
As a side effect this might be seen as a template for further `CUDA` projects using `cmake`. To understand better it is useful to look at the corresponding profiles in `NCU`.

Execute 

```
mkdir build && cd build
```

Followed by

```
cmake ..
```

and than build all the kernels

```
cmake --build .
```

You may than create a profile like so (when in the `build` directory):

For the steps below make sure you create a `profiles` directory in the root folder.

```
ncu --set full -o ../profiles/stream_1 ./main_stream 0
```

or simply run the kernels.

For different architecture or arguments to the compiler adjust `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.
Note the outcommented code using `atomicAdd` on `float4` will only work on Hopper (it's not performant though).

You can see [here](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#l2-cache) for the corresponding section in Cuda best practices guide. 
