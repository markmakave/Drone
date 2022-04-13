#pragma once

#include <cmath>

#include "map.h"

namespace lm {

///////////////////////////////////////////////////////////////////////////////////////////////////

static float gaussian_distribudion(int x, int y, float dispersion) {
    return 1.f / (2.f * M_PI * dispersion * dispersion) * exp(-(x * x + y * y) / (2.f * dispersion * dispersion));
}

map<float> gaussian(int radius, float dispersion = 1.f) {
    map<float> core(radius * 2 + 1, radius * 2 + 1);

    for (int x = 0; x < core.width(); ++x) {
        for (int y = 0; y < core.height(); ++y) {
            core(x, y) = gaussian_distribudion(x - radius, y - radius, dispersion);
        }
    }

    return core;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

map<float> sharpness() {
    map<float> core(3, 3);

    core(0, 0) = -1.f;
    core(1, 0) = -1.f;
    core(2, 0) = -1.f;

    core(0, 1) = -1.f;
    core(1, 1) = 9.f;
    core(2, 1) = -1.f;

    core(0, 2) = -1.f;
    core(1, 2) = -1.f;
    core(2, 2) = -1.f;

    return core;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

map<float> edges() {
    map<float> core(3, 3);

    core(0, 0) = -1.f;
    core(1, 0) = -1.f;
    core(2, 0) = -1.f;

    core(0, 1) = -1.f;
    core(1, 1) = 8.f;
    core(2, 1) = -1.f;

    core(0, 2) = -1.f;
    core(1, 2) = -1.f;
    core(2, 2) = -1.f;

    return core;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

map<float> vsobel() {
    map<float> core(3, 3);

    core(0, 0) = -1.f;
    core(1, 0) = -2.f;
    core(2, 0) = -1.f;

    core(0, 1) = 0.f;
    core(1, 1) = 0.f;
    core(2, 1) = 0.f;

    core(0, 2) = 1.f;
    core(1, 2) = 2.f;
    core(2, 2) = 1.f;

    return core;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

map<float> hsobel() {
    map<float> core(3, 3);

    core(0, 0) = -1.f;
    core(1, 0) = 0.f;
    core(2, 0) = 1.f;

    core(0, 1) = -2.f;
    core(1, 1) = 0.f;
    core(2, 1) = 2.f;

    core(0, 2) = -1.f;
    core(1, 2) = 0.f;
    core(2, 2) = 1.f;

    return core;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}