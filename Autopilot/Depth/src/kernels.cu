#include <cstring>

#include "kernels.h"
#include "map.h"

using lumina::map;

__device__ uint8_t convert_disparity(int infimum, int center, int width) {
    return UINT8_MAX * abs(center - infimum) / float(width);
}

__global__ void lumina::depth(map<uint8_t>* left, map<uint8_t>* right, map<uint8_t>* result, int radius, int thresold) {

    int16_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int16_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < radius || y < radius || x >= result->width() - radius - 1 || y >= result->height() - radius - 1)
        return;

    int infimum_position = x;
    int infimum_value = UINT8_MAX * (radius + radius + 1) * (radius + radius + 1);

    for (int center = radius; center < x; ++center) {
        int16_t cur_difference;
        int16_t sum_difference = 0;

        for (int x_offset = -radius; x_offset <= radius; ++x_offset) {
            for (int y_offset = -radius; y_offset <= radius; ++y_offset) {
                cur_difference = (int16_t)(*left)(x + x_offset, y + y_offset) - (int16_t)(*right)(center + x_offset, y + y_offset);
                sum_difference += abs(cur_difference);
            }
        }

        if (sum_difference < infimum_value) {
            infimum_position = center;
            infimum_value = sum_difference;
        }
    }

    (*result)(x, y) = convert_disparity(infimum_position, x, result->width());
}

__device__ int8_t core[3][3] = {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}};

__global__ void lumina::filter(map<uint8_t>* in, map<uint8_t>* out) {
    int16_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int16_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < 1 || y < 1 || x >= out->width() - 2 || y >= out->height() - 2) return;

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

__device__ static void sort(uint8_t* arr, int n) {
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

__global__ void lumina::median(map<uint8_t>* in, map<uint8_t>* out) {
    int16_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int16_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < MEDIAN_RADIUS || y < MEDIAN_RADIUS || x >= out->width() - MEDIAN_RADIUS - 1 || y >= out->height() - MEDIAN_RADIUS - 1) return;

    uint8_t arr[(MEDIAN_RADIUS * 2 + 1) * (MEDIAN_RADIUS * 2 + 1)];
    for (int _x = -MEDIAN_RADIUS; _x <= MEDIAN_RADIUS; ++_x) {
        for (int _y = -MEDIAN_RADIUS; _y <= MEDIAN_RADIUS; ++_y) {
            arr[(_y + MEDIAN_RADIUS) * (2 * MEDIAN_RADIUS + 1) + (_x + MEDIAN_RADIUS)] = (*in)(x + _x, y + _y);
        }
    }

    sort(arr, (MEDIAN_RADIUS * 2 + 1) * (MEDIAN_RADIUS * 2 + 1));

    (*out)(x, y) = arr[(MEDIAN_RADIUS * 2 + 1) * (MEDIAN_RADIUS * 2 + 1) / 2];
}