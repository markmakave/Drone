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

    lumina::Camera camera(0, WIDTH, HEIGHT);

    lumina::map<rgba> frame(WIDTH, HEIGHT);
    cv::Mat image(HEIGHT, WIDTH, CV_8UC4, &frame[0]);

    while(1) {
        camera >> frame;

        cv::imshow("frame", image);
        cv::waitKey(1);
    }

    return 0;
}