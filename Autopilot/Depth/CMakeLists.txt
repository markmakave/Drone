cmake_minimum_required(VERSION 3.21)

project(depth LANGUAGES C CXX CUDA)

SET(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-g")

find_package(Iconv  REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(/usr/local/cuda/include)

add_executable(depth
    ./src/main.cpp
    ./src/stereobm.cpp
    ./src/kernels.cu
)

link_directories(/usr/local/cuda/lib64)
target_link_libraries(depth ${OpenCV_LIBS})
link_libraries(depth cudart)