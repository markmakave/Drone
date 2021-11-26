project(drone LANGUAGES C CXX CUDA ASM)

find_package(OpenCV REQUIRED)

include_directories(include)

set(CMAKE_CXX_FLAGS "-g")

add_executable(drone
	src/main.cu

	src/depth/depth.cu
	src/device/camera/camera.cpp
)

target_link_libraries(drone ${OpenCV_LIBS})