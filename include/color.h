#pragma once
#include <cstdint>
#include <cuda_runtime.h>

class rgba {
public:
    uint8_t r, g, b, a;

    __host__ __device__ rgba() {
        r = 0;
        g = 0;
        b = 0;
        a = 0;
    }
    __host__ __device__ rgba(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a = 255) {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }
};