cmake_minimum_required(VERSION 3.24)
project(damendaemon LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(damendaemon-cuda-probe native/cuda_probe.cu)
target_compile_features(damendaemon-cuda-probe PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(damendaemon-cuda-probe PRIVATE CUDA::cudart CUDA::cublas)
# Pass -DCMAKE_CUDA_ARCHITECTURES=86 for the initial NVIDIA A2 target.

add_library(damendaemon-q4 SHARED native/q4_gemv.cu)
target_compile_features(damendaemon-q4 PRIVATE cuda_std_17)
target_link_libraries(damendaemon-q4 PRIVATE CUDA::cudart)
target_compile_options(damendaemon-q4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_library(damendaemon-kquant SHARED native/kquant_gemv.cu)
target_compile_features(damendaemon-kquant PRIVATE cuda_std_17)
target_link_libraries(damendaemon-kquant PRIVATE CUDA::cudart)
target_compile_options(damendaemon-kquant PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_library(damendaemon-kquant-residual SHARED native/kquant_residual.cu)
target_compile_features(damendaemon-kquant-residual PRIVATE cuda_std_17)
target_link_libraries(damendaemon-kquant-residual PRIVATE CUDA::cudart)
target_compile_options(damendaemon-kquant-residual PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_library(damendaemon-kquant-residual-f32 SHARED native/kquant_residual_f32.cu)
target_compile_features(damendaemon-kquant-residual-f32 PRIVATE cuda_std_17)
target_link_libraries(damendaemon-kquant-residual-f32 PRIVATE CUDA::cudart)
target_compile_options(damendaemon-kquant-residual-f32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_library(damendaemon-q4-residual SHARED native/q4_residual.cu)
target_compile_features(damendaemon-q4-residual PRIVATE cuda_std_17)
target_link_libraries(damendaemon-q4-residual PRIVATE CUDA::cudart)
target_compile_options(damendaemon-q4-residual PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

option(DAMENDAEMON_BUILD_AWQ_MARLIN "Build optional native Marlin AWQ kernels" OFF)
if(DAMENDAEMON_BUILD_AWQ_MARLIN OR DAMENDAEMON_BUILD_AWQ_MARLIN_EXPERIMENT)
    add_library(damendaemon-awq-marlin SHARED native/awq_marlin.cu)
    target_include_directories(damendaemon-awq-marlin PRIVATE third_party/marlin)
    target_compile_features(damendaemon-awq-marlin PRIVATE cuda_std_17)
    target_link_libraries(damendaemon-awq-marlin PRIVATE CUDA::cudart)
    target_compile_options(damendaemon-awq-marlin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

option(DAMENDAEMON_BUILD_Q4_MARLIN "Build optional native Marlin Q4_0 kernels" OFF)
if(DAMENDAEMON_BUILD_Q4_MARLIN)
    add_library(damendaemon-q4-marlin SHARED native/q4_marlin.cu)
    target_include_directories(damendaemon-q4-marlin PRIVATE third_party/marlin)
    target_compile_features(damendaemon-q4-marlin PRIVATE cuda_std_17)
    target_link_libraries(damendaemon-q4-marlin PRIVATE CUDA::cudart)
    target_compile_options(damendaemon-q4-marlin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()
