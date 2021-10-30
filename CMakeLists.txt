project(drone LANGUAGES C CXX CUDA ASM)

set(CMAKE_CXX_FLAGS "-O3 -fsanitize=undefined")

add_executable(drone
	src/main.cpp
)