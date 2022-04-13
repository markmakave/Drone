#include <iostream>

#include <png++/png.hpp>

#include "map.h"
#include "image.h"
#include "stereobm.h"

#define DATASET "/home/lumina/dev/lumina/dataset/"

#define FOCAL_LENGTH 10.0
#define CAMERA_DISTANCE 200.0
#define MEDIAN_BLOCK_SIZE 2
#define DISPARITY_BLOCK_SIZE 5
#define DISPARITY_THRESHOLD 50

int main() {

    lm::Image<lm::grayscale> left(DATASET "left.png"), right(DATASET "right.png");
    lm::map<float> depth;

    lm::cuda::StereoBM matcher(
        FOCAL_LENGTH,
        CAMERA_DISTANCE,
        MEDIAN_BLOCK_SIZE,
        DISPARITY_BLOCK_SIZE,
        DISPARITY_THRESHOLD
    );

    matcher.compute(left, right, depth);

    lm::Image<lm::rgba>(lm::gradient(depth)).save("image.png");

    return 0;
}