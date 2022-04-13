project(convolve LANGUAGES C CXX CUDA)

include_directories(~/dev/lumina/include)
include_directories(/usr/local/cuda/include)

set(CMAKE_C_FLAGS "-g")
set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_CUDA_FLAGS "-g")

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

add_executable(convolve
    main.cpp
    ~/dev/lumina/src/stereobm.cpp
    ~/dev/lumina/src/kernels.cu
)

link_directories(/usr/local/cuda/lib64)
link_libraries(convolve cudart)

target_link_libraries(convolve ${PNG_LIBRARY})