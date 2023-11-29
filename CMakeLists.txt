cmake_minimum_required(VERSION 3.26)
project(Genre_Reveal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "-static")
find_package(MPI REQUIRED)

add_executable(serial serial.cpp
        point.h)

add_executable(shared_cpu shared_cpu.cpp
        point.h)

add_executable(dist_cpu dist_cpu.cpp
        point.h)


target_link_libraries(dist_cpu PUBLIC MPI::MPI_CXX)