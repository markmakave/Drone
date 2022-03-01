#include <iostream>
#include <fstream>
#include <iomanip>

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

double tempreature(THERMAL_DEVICE id) {
    std::ifstream file(std::string("/sys/class/thermal/thermal_zone") + std::to_string(id) + "/temp");
    double temp;
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
            // double cpu_temp = tempreature(CPU);
            // double gpu_temp = tempreature(GPU);
            // std::cout << std::fixed << std::setprecision(1);
            // std::cout << "CPU tempreature: " << cpu_temp << std::endl;
            // std::cout << "GPU tempreature: " << gpu_temp << std::endl;
            matcher.compute(left_frame, right_frame, depth_map);
        }

        // cv::Mat image(depth_map.height(), depth_map.width(), CV_8UC1, depth_map.data());
        // cv::resize(image, scale, cv::Size(640, 480));
        // cv::imshow("depth", scale);
        // cv::waitKey(1);
    }

    return 0;
}