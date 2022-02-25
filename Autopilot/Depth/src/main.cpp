#include <iostream>

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

int main(int argc, char** argv) {

    // INIT
    lumina::Camera left_camera(1, WIDTH, HEIGHT), right_camera(2, WIDTH, HEIGHT);
    lumina::map<uint8_t> left_frame, right_frame, depth_map;
    lumina::StereoBM matcher(5, 0);

    // LOOP

    while (1) {
        left_camera  >> left_frame;
        right_camera >> right_frame;

        {
            Timer timer;
            matcher.compute(left_frame, right_frame, depth_map);
        }

        // cv::Mat image(depth_map.height(), depth_map.width(), CV_8UC1, depth_map.data());
        // cv::imshow("depth", image);
        // cv::waitKey(1);
    }

    return 0;
}