cmake_minimum_required(VERSION 3.21)
project(depth LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-g")

include_directories(include)
include_directories(/usr/local/cuda/include)

add_executable(depth
    ./src/main.cpp
    ./src/stereobm.cpp
    ./src/kernels.cu
)

link_directories(/usr/local/cuda/lib64)
link_libraries(depth cudart)
