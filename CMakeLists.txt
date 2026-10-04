cmake_minimum_required(VERSION 3.24)
project(dllm LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(dllm-cuda-probe native/cuda_probe.cu)
target_compile_features(dllm-cuda-probe PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(dllm-cuda-probe PRIVATE CUDA::cudart CUDA::cublas)
# Pass -DCMAKE_CUDA_ARCHITECTURES=86 for the initial NVIDIA A2 target.

add_library(dllm-q4 SHARED native/q4_gemv.cu)
target_compile_features(dllm-q4 PRIVATE cuda_std_17)
target_link_libraries(dllm-q4 PRIVATE CUDA::cudart)
target_compile_options(dllm-q4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
