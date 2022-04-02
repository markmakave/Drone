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
#include <cstdlib>

// #define DEBUG

enum {
    WIDTH = 160, 
    HEIGHT = 120,
    FOCAL_LENGTH = 40,
    CAMERA_DISTANCE = 157,
    BLOCK_SIZE = 3,
    DISTINCTION_THRESHOLD = 10,
    VALIDATION_THRESHOLD = 10
};

int main(int argc, char** argv) {

    // INIT
    
    lm::autopilot::Camera left_camera(1, WIDTH, HEIGHT),
                          right_camera(0, WIDTH, HEIGHT);

    lm::autopilot::StereoBM matcher(FOCAL_LENGTH,
                                    CAMERA_DISTANCE,
                                    BLOCK_SIZE,
                                    DISTINCTION_THRESHOLD,
                                    VALIDATION_THRESHOLD);

    lm::autopilot::Environment env(WIDTH, HEIGHT, 90.f);

    lm::map<lm::grayscale> left_frame, right_frame;
    lm::map<float> depth_map;

    // LOOP

    while (1) 
    {
        left_camera  >> left_frame;
        right_camera >> right_frame;

        matcher.compute(left_frame, right_frame, depth_map);
        
        lm::Image(lm::gradient(depth_map, 0, 250)).save("image.png");
    }

    env.apply(depth_map);
    env.export_stl();

    return 0;
}
