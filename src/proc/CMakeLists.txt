project(proc LANGUAGES CUDA CXX)

find_package(OpenCV REQUIRED)

include_directories(~/dev/drone/include)

add_executable(proc
    main.cu
    proc.cu
    ~/dev/drone/src/device/camera/camera.cpp
)

target_link_libraries(proc ${OpenCV_LIBS})