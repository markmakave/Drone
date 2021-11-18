#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "camera.h"

#define ITER 10000

int main() {

    Camera camera_right(0), camera_left(1);

    camera_right.start();
    camera_left.start();

    auto start = std::chrono::high_resolution_clock::now();

    const char str[] = 
    "aaa\n
    aaa0";

    cv::Mat left(120, 160, CV_8UC1), right(120, 160, CV_8UC1);
    for (int i = 0; i < ITER; ++i) {
        camera_right.capture(right.data);
        camera_left.capture(left.data);

        cv::imshow("left", left);
        cv::imshow("right", right);
        cv::waitKey(1);
    }

    auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << "Capture took " <<  microseconds / 1000000.f << " secs" << std::endl;

    camera_right.stop();
    camera_left.stop();

    return 0;
}