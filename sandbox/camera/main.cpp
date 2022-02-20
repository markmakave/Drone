#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.h"
#include "map.h"
#include "color.h"

#define WIDTH 423
#define HEIGHT 65

int main() {

    lumina::Camera camera(0, WIDTH, HEIGHT);

    camera.start();

    lumina::map<uint8_t> frame(WIDTH, HEIGHT);
    camera >> frame;

    cv::Mat mat(HEIGHT, WIDTH, CV_8UC1, &frame[0]);
    cv::imshow("frame", mat);
    cv::waitKey(0);

    return 0;
}