#include <iostream>
#include <chrono>

#include <cstdlib>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.h"
#include "map.h"
#include "color.h"

#define WIDTH 160
#define HEIGHT 120

int main() {

    lumina::Camera camera(2, WIDTH, HEIGHT);
    lumina::map<uint8_t> frame1, frame2;

    while(1) {
        camera  >> frame1 >> frame2;
    }

    return 0;
}