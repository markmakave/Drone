#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "proc.h"
#include "map.h"
#include "camera.h"

int main() {

    map<int8_t> filter(3,3);
    filter(0, 0) = -1;
    filter(0, 1) = -1;
    filter(0, 2) = -1;

    filter(1, 0) = -1;
    filter(1, 1) = 9;
    filter(1, 2) = -1;

    filter(2, 0) = -1;
    filter(2, 1) = -1;
    filter(2, 2) = -1;

    Camera cam(0);
    cam.start();

    

    map<uint8_t> raw(160, 120);
    map<uint8_t> filtered(160, 120);

    cv::Mat expand(480, 640, CV_8UC1);
    while (1) {
        cam.capture(raw.host_data);
        
        sobel(raw, filtered, filter);

        cv::Mat frame(120, 160, CV_8UC1, filtered.host_data);
        cv::resize(frame, expand, cv::Size(640, 480));

        cv::imshow("frame", expand);
        cv::waitKey(1);
    }

    cam.stop();

    return 0;
}