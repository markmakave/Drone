#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "stereobm.h"
#include "camera.h"
#include "map.h"
#include "timer.h"

enum {
    WIDTH = 160, 
    HEIGHT = 120 
};

enum THERMAL_DEVICE {
    A0   = 0,
    CPU  = 1,
    GPU  = 2,
    PLL  = 3,
    PMIC = 4,
    FAN  = 5
};

float tempreature(THERMAL_DEVICE id) {
    std::ifstream file(std::string("/sys/class/thermal/thermal_zone") + std::to_string(id) + "/temp");
    if (!file) {
        throw std::runtime_error("Tempreature device not recognized");
    }
    float temp;
    file >> temp;
    file.close();
    return temp / 1000;
}

int main(int argc, char** argv) {

    // INIT

    lumina::Camera left_camera(1, WIDTH, HEIGHT), right_camera(2, WIDTH, HEIGHT);
    lumina::map<uint8_t> left_frame, right_frame, depth_map;
    lumina::StereoBM matcher(3, 0);
    cv::Mat scale;

    // LOOP

    while (1) {
        Timer timer;
        left_camera  >> left_frame;
        right_camera >> right_frame;

        {
            matcher.compute(left_frame, right_frame, depth_map);
        }

        // cv::Mat image(depth_map.height(), depth_map.width(), CV_8UC1, depth_map.data());
        // cv::resize(image, scale, cv::Size(640, 480));
        // cv::imshow("depth", scale);
        // cv::waitKey(1);
    }

    return 0;
}