#include <iostream>
#include <cstring>

#include "map.h"

#define WIDTH 160
#define HEIGHT 120

#define ITERATIONS 10000000

int main() {

    lumina::map<uint8_t> first(WIDTH), second(HEIGHT);

    for (long long i = 0; i < ITERATIONS; ++i) {
        std::memcpy(second.data(), first.data(), first.size());
    }

    return 0;
}

