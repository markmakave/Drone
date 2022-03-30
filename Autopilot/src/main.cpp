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

#include <cstdlib>

// #define DEBUG

enum {
    WIDTH = 160, 
    HEIGHT = 120,
    FOCAL_LENGTH = 10,
    DISTANCE = 220
};

#ifdef DEBUG

void* operator new(size_t size) {
    void* ptr = malloc(size);
    std::cout << "Allocating " << size << " bytes at " << ptr << std::endl;
    return ptr;
}

void operator delete(void* ptr) {
    std::cout << "Dellocating " << ptr << std::endl;
    return free(ptr);
}

#endif

int main(int argc, char** argv) {

    // INIT
    
    lm::autopilot::Camera left_camera(1, WIDTH, HEIGHT), right_camera(0, WIDTH, HEIGHT);
    lm::autopilot::StereoBM matcher(FOCAL_LENGTH, DISTANCE);
    lm::autopilot::Environment env(WIDTH, HEIGHT, 90.f);

    lm::map<uint8_t> left_frame, right_frame;
    lm::map<float> depth_map;

    // LOOP

    // while (1) 
    {
        left_camera  >> left_frame;
        right_camera >> right_frame;

        {
            matcher.compute(left_frame, right_frame, depth_map);
        }

        
        lm::Image(lm::gradient(depth_map)).save("image.png");
    }

    env.apply(depth_map);
    env.export_stl();

    return 0;
}
