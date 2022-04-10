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
#include "timer.h"

#define LM_CUDA

void info();

enum {
    WIDTH = 640, 
    HEIGHT = 480,

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

    lm::Camera left_camera(1, WIDTH, HEIGHT),
               right_camera(0, WIDTH, HEIGHT);

    left_camera.info();
    right_camera.info();


    #ifdef LM_CUDA
    lm::cuda::StereoBM matcher
    #else
    lm::StereoBM matcher
    #endif
    (
        FOCAL_LENGTH,
        CAMERA_DISTANCE,
        BLOCK_SIZE,
        DISTINCTION_THRESHOLD,
        VALIDATION_THRESHOLD
    );

    lm::map<lm::grayscale> left_frame, right_frame;
    lm::map<float> depth_map;

    // LOOP

    while (1) 
    {
        left_camera  >> left_frame;
        right_camera >> right_frame;

        {
            Timer timer;
            matcher.compute(left_frame, right_frame, depth_map);
        }
        
        lm::Image(lm::gradient(depth_map, GRADIENT_FLOOR, GRADIENT_ROOF)).save("image.png");
    }

    return 0;
}
