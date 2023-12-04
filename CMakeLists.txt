cmake_minimum_required(VERSION 3.26)
project(Genre_Reveal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "-static")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
find_package(MPI REQUIRED)

add_executable(serial serial.cpp
        point.h)

add_executable(shared_cpu shared_cpu.cpp
        point.h)

add_executable(dist_cpu dist_cpu.cpp
        point.h)


target_link_libraries(dist_cpu PUBLIC MPI::MPI_CXX)

# Clion has a weird workflow for CUDA on windows. Because you have to use the visual studio toolchain, other
# implementations run much slower. Comment/uncomment this line to exclude/include GPU implementations:
add_subdirectory("${PROJECT_SOURCE_DIR}/cuda" "${PROJECT_SOURCE_DIR}/cuda_output")