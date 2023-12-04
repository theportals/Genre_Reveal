cmake_minimum_required(VERSION 3.26)
project(Genre_Reveal_Cuda LANGUAGES CXX CUDA)

find_package(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS "-static")
set(CMAKE_CUDA_ARCHITECTURES 70)

add_executable(shared_gpu shared_gpu.cu
        ../point.h
)

set_target_properties(shared_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
