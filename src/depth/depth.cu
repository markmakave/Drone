#include "depth.h"

#define RADIUS 2
#define THRESOLD 400 * (RADIUS * 2 + 1) * (RADIUS * 2 + 1)
#define MULTIPLIER 1

#define WIDTH 160
#define HEIGHT 120

__device__ inline uint8_t disparity2depth(uint8_t disparity) {
    return (disparity / (float)WIDTH) * UINT8_MAX * MULTIPLIER;
}

__global__ void depth(map<uint8_t>* left, map<uint8_t>* right, map<uint8_t>* result) {

    int16_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int16_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < RADIUS || y < RADIUS || x >= result->width - RADIUS - 1 || y >= result->height - RADIUS - 1) return;

    uint8_t hit = x;
    uint64_t delta = UINT64_MAX;

    for (uint16_t i = RADIUS; i < x; ++i) {
        
        uint64_t cur = 0;
        for (int8_t _x = -RADIUS; _x <=  RADIUS; ++_x) {
            for (int8_t _y = -RADIUS; _y <= RADIUS; ++_y) {
                int32_t difference = (int16_t)(*left)(x + _x, y + _y) - (int16_t)(*right)((int16_t)i + _x, y + _y);
                cur += difference * difference;
            }
        }

        if (cur < delta && cur < THRESOLD) {
            hit = i;
            delta = cur;
        }
    }

    //(*result)(x, y) = (x - hit) * MULTIPLIER;
    (*result)(x, y) = disparity2depth(x - hit);
}