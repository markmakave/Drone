#pragma once
#include <cstdint>

struct rgba {
    uint8_t r, g, b, a;

    rgba(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 255)
        : r(r), g(g), b(b), a(a) {
    }
};