#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "stereobm.h"
#include "camera.h"
#include "map.h"
#include "environment.h"
#include "geometry.h"

enum {
    WIDTH = 160, 
    HEIGHT = 120 
};

int main(int argc, char** argv) {

    // INIT
    
    lumina::Camera left_camera(0, WIDTH, HEIGHT), right_camera(1, WIDTH, HEIGHT);
    lumina::map<uint8_t> left_frame, right_frame, depth_map;
    lumina::StereoBM matcher(3, 0);

    // LOOP

    //while (1) 
    {
        left_camera  >> left_frame;
        right_camera >> right_frame;

        {
            matcher.compute(left_frame, right_frame, depth_map);
        }

        for (size_t x = 0; x < depth_map.width(); ++x) {
            for (size_t y = 0; y < depth_map.height(); ++y) {

                lumina::dim dot;

            }
        }
    }

    return 0;
}
