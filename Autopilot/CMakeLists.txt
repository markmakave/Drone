cmake_minimum_required(VERSION 3.21)
project(Autopilot LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-g")

include_directories(include)
include_directories(/usr/local/cuda/include)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

add_executable(Autopilot
    ./src/main.cpp
    ./src/stereobm.cpp
    ./src/kernels.cu
)

link_directories(/usr/local/cuda/lib64)
link_libraries(Autopilot cudart)

target_link_libraries(Autopilot ${PNG_LIBRARY})
