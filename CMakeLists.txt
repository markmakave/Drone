project(drone LANGUAGES C CXX CUDA ASM)

find_package(OpenCV REQUIRED)

include_directories(include)

set(CMAKE_CUDA_FLAGS "-fomit-frame-pointer -O3 -Xptxas -O3")

add_executable(drone
	src/main.cu

	src/depth/depth.cu
)

target_link_libraries(drone ${OpenCV_LIBS})