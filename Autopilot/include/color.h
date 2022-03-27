#pragma once
#include <cstdint>

uint8_t clamp(int c) {
    if (c > 255)
        return 255;
    if (c < 0)
        return 0;
    return c;
}

struct yuyv {
    uint8_t y, u, v;

    yuyv(uint8_t y = 0, uint8_t u = 0, uint8_t v = 0)
        : y(y), u(u), v(v) {
    }
};

struct rgba {
    uint8_t r, g, b, a;

    rgba(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 255)
        : r(r), g(g), b(b), a(a) {
    }

    rgba(const yuyv& color) {
        r = clamp((int)color.y + (1.732446 * ((int)color.u - 128)));
        b = clamp((int)color.y + (1.370705 * ((int)color.v - 128)));
        g = clamp((int)color.y - (0.698001 * ((int)color.v - 128)) - (0.337633 * ((int)color.u - 128)));
        a = 255;
    }

    operator yuyv() {
        return yuyv();
    }
};