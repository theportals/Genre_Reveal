cmake_minimum_required(VERSION 3.26)
project(Genre_Reveal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(serial serial.cpp)
