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

add_library(dllm-kquant SHARED native/kquant_gemv.cu)
target_compile_features(dllm-kquant PRIVATE cuda_std_17)
target_link_libraries(dllm-kquant PRIVATE CUDA::cudart)
target_compile_options(dllm-kquant PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_library(dllm-kquant-residual SHARED native/kquant_residual.cu)
target_compile_features(dllm-kquant-residual PRIVATE cuda_std_17)
target_link_libraries(dllm-kquant-residual PRIVATE CUDA::cudart)
target_compile_options(dllm-kquant-residual PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_library(dllm-kquant-residual-f32 SHARED native/kquant_residual_f32.cu)
target_compile_features(dllm-kquant-residual-f32 PRIVATE cuda_std_17)
target_link_libraries(dllm-kquant-residual-f32 PRIVATE CUDA::cudart)
target_compile_options(dllm-kquant-residual-f32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_library(dllm-q4-residual SHARED native/q4_residual.cu)
target_compile_features(dllm-q4-residual PRIVATE cuda_std_17)
target_link_libraries(dllm-q4-residual PRIVATE CUDA::cudart)
target_compile_options(dllm-q4-residual PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
