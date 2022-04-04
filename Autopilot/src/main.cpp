#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "stereobm.h"
#include "camera.h"
#include "map.h"
#include "environment.h"
#include "geometry.h"
#include "color.h"
#include "image.h"

void info();

enum {
    WIDTH = 160, 
    HEIGHT = 120,

    FOCAL_LENGTH = 40,
    CAMERA_DISTANCE = 157,

    BLOCK_SIZE = 3,
    DISTINCTION_THRESHOLD = 3,
    VALIDATION_THRESHOLD = 1,

    GRADIENT_FLOOR = 0,
    GRADIENT_ROOF = 150
};

int main(int argc, char** argv) {

    // INIT

    //info();

    lm::autopilot::Camera left_camera(1, WIDTH, HEIGHT),
                          right_camera(0, WIDTH, HEIGHT);

    left_camera.info();
    right_camera.info();

    lm::autopilot::StereoBM matcher(FOCAL_LENGTH,
                                    CAMERA_DISTANCE,
                                    BLOCK_SIZE,
                                    DISTINCTION_THRESHOLD,
                                    VALIDATION_THRESHOLD);

    lm::autopilot::Environment env(WIDTH, HEIGHT, 90.f);

    lm::map<lm::grayscale> left_frame, right_frame;
    lm::map<float> depth_map;

    // LOOP

    // while (1) 
    {
        left_camera  >> left_frame;
        right_camera >> right_frame;

        matcher.compute(left_frame, right_frame, depth_map);
        
        lm::Image(lm::gradient(depth_map, GRADIENT_FLOOR, GRADIENT_ROOF)).save("image.png");
    }

    env.apply(depth_map);
    env.export_stl();

    return 0;
}

void info() {

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device name:                %s\n", deviceProp.name); 
    printf("Major revision number:      %d\n", deviceProp.major);
    printf("Minor revision Number:      %d\n", deviceProp.minor); 
    printf("Total Global Memory:        %lu\n", deviceProp.totalGlobalMem);
    printf("Total shared mem per block: %lu\n", deviceProp.sharedMemPerBlock); 
    printf("Total const mem size:       %lu\n", deviceProp.totalConstMem); 
    printf("Warp size:                  %d\n", deviceProp.warpSize); 
    printf("Maximum block dimensions:   %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]); 
    printf("Maximum grid dimensions:    %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]); 
    printf("Clock Rate:                 %d\n", deviceProp.clockRate); 
    printf("Number of muliprocessors:   %d\n", deviceProp.multiProcessorCount); 
}