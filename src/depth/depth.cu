#include "depth.h"

#define RADIUS 1
#define THRESOLD 100 * (RADIUS * 2 + 1) * (RADIUS * 2 + 1)
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

        if (cur < delta /*&& cur < THRESOLD*/) {
            hit = i;
            delta = cur;
        }
    }

    (*result)(x, y) = disparity2depth(x - hit);
}

__device__ int8_t core[3][3] = {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}};

__global__ void filter(map<uint8_t>* in, map<uint8_t>* out) {
    int16_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int16_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < 1 || y < 1 || x >= out->width - 2 || y >= out->height - 2) return;

    int16_t sum = 0;
    for (int _x = -1; _x <= 1; ++_x) {
        for (int _y = -1; _y <= 1; ++_y) {
            sum += (int16_t)core[_y + 1][_x + 1] * (int16_t)(*in)(x + _x, y + _y);
        }
    }

    if (sum > 255) {
        sum = 255;
    } else if (sum < 0) {
        sum = 0;
    }


    (*out)(x, y) = sum;
}

#define MEDIAN_RADIUS 2

__device__ void sort(uint8_t* arr, int n) {
    for (int i = 0; i < n; i++) {
        int minPosition = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[minPosition] > arr[j])
                minPosition = j;
        }
        uint8_t tmp = arr[minPosition];
        arr[minPosition] = arr[i];
        arr[i] = tmp;
    }
}

__global__ void median(map<uint8_t>* in, map<uint8_t>* out) {
    int16_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int16_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < MEDIAN_RADIUS || y < MEDIAN_RADIUS || x >= out->width - MEDIAN_RADIUS - 1 || y >= out->height - MEDIAN_RADIUS - 1) return;

    uint8_t arr[(MEDIAN_RADIUS * 2 + 1) * (MEDIAN_RADIUS * 2 + 1)];
    for (int _x = -MEDIAN_RADIUS; _x <= MEDIAN_RADIUS; ++_x) {
        for (int _y = -MEDIAN_RADIUS; _y <= MEDIAN_RADIUS; ++_y) {
            arr[(_y + MEDIAN_RADIUS) * (2 * MEDIAN_RADIUS + 1) + (_x + MEDIAN_RADIUS)] = (*in)(x + _x, y + _y);
        }
    }

    sort(arr, (MEDIAN_RADIUS * 2 + 1) * (MEDIAN_RADIUS * 2 + 1));

    (*out)(x, y) = arr[(MEDIAN_RADIUS * 2 + 1) * (MEDIAN_RADIUS * 2 + 1) / 2];
}