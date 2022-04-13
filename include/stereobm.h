#pragma once
#include <cstdint>

#include <cuda_runtime.h>

#include "map.h"
#include "cuda_map.h"
#include "color.h"

namespace lm {

class StereoBM {

    float focal_length;
    float camera_distance;
    int median_block_size;
    int disparity_block_size;
    int disparity_threshold;

public:

    StereoBM(
        float focal_length,
        float camera_distance,
        int   median_block_size    = 5,
        int   disparity_block_size = 5,
        int   disparity_threshold  = 10);

    void compute(
        const map<grayscale> &left_frame,
        const map<grayscale> &right_frame,
              map<float>     &depth_map);

};

namespace cuda {

class StereoBM {

    float focal_length;
    float camera_distance;
    int median_block_size;
    int disparity_block_size;
    int disparity_threshold;

    map<grayscale> cuda_left_frame, cuda_right_frame;
    map<grayscale> cuda_left_median, cuda_right_median;
    map<int>       cuda_disparity;
    map<float>     cuda_depth;
    
public:

    StereoBM(
        float focal_length,
        float camera_distance,
        int   median_block_size    = 5,
        int   disparity_block_size = 5,
        int   disparity_threshold  = 10);

    void compute(
        const lm::map<grayscale> &left_frame,
        const lm::map<grayscale> &right_frame,
              lm::map<float>     &depth_map);

private:

    void _update(int width, int height);

};

}

}