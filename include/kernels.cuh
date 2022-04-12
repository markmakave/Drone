#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "map.h"
#include "color.h"
#include "stereobm.h"

namespace lm {

namespace cuda {

__global__ void disparity(const map<uint8_t>* left,
                          const map<uint8_t>* right,
                                map<int>* disparity,
                          const int block_radius,
                          const int threshold);

__global__ void depth(const map<int>* disparity,
                            map<float>* depth,
                      const float focal_lenght,
                      const float camera_distance);

__global__ void convolve(const map<grayscale> *input,
                         const map<float> *core,
                               map<grayscale> *output);

}

}