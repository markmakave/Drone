#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "map.h"

namespace lumina {

    __global__ void depth(map<uint8_t>* left, map<uint8_t>* right, map<uint8_t>* result, int radius, int thresold);

    __global__ void filter(map<uint8_t>*, map<uint8_t>*);

    __global__ void median(map<uint8_t>*, map<uint8_t>*);

}