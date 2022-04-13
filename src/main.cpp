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

enum {
    WIDTH = 160, 
    HEIGHT = 120,

    FOCAL_LENGTH = 40,
    CAMERA_DISTANCE = 157,

    MEDIAN_BLOCK_SIZE = 5,

    BLOCK_SIZE = 10,
    THRESHOLD = 2000,

    GRADIENT_FLOOR = 0,
    GRADIENT_ROOF = 500
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
        MEDIAN_BLOCK_SIZE,
        BLOCK_SIZE,
        THRESHOLD
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
