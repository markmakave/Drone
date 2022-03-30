#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "map.h"

namespace lm {

namespace autopilot {

__global__ void depth(map<uint8_t>* left, map<uint8_t>* right, map<float>* result, int radius, int thresold, float focal_length, float distance);

__global__ void filter(map<uint8_t>*, map<uint8_t>*);

__global__ void median(map<uint8_t>*, map<uint8_t>*);

}

}