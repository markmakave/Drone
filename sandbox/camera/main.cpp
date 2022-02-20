#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.h"
#include "map.h"
#include "color.h"

int main() {

    lumina::Camera camera(0, 640, 480);

    camera.start();

    lumina::map<rgba> frame(640, 480);
    camera >> frame;

    cv::Mat mat(480, 640, CV_8SC1, &frame[0]);
    cv::imshow("frame", mat);
    cv::waitKey(0);

    return 0;
}