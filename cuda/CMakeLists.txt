cmake_minimum_required(VERSION 3.26)
project(Genre_Reveal_Cuda LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_CUDA_ARCHITECTURES 70)

add_executable(shared_gpu shared_gpu.cu
        ../point.h
)

add_executable(dist_gpu dist_gpu.cpp
        updateCentroids.cu
        ../point.h
)

target_link_libraries(dist_gpu PUBLIC MPI::MPI_CXX)
set_target_properties(shared_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
