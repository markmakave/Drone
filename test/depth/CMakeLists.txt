cmake_minimum_required(VERSION 3.21)
project(depth LANGUAGES C CXX CUDA)

include_directories(~/dev/lumina/include)
include_directories(/usr/local/cuda/include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_C_FLAGS "-g -fopenmp")
set(CMAKE_CXX_FLAGS "-g -fopenmp")
set(CMAKE_CUDA_FLAGS "")

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

add_executable(depth 
    main.cpp
    ~/dev/lumina/src/stereobm.cpp
    ~/dev/lumina/src/kernels.cu
)

link_directories(/usr/local/cuda/lib64)
link_libraries(depth cudart)

target_link_libraries(depth ${PNG_LIBRARY})
