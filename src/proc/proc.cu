#include "proc.h"

void sobel(map<uint8_t> &in, map<uint8_t> &out, map<int8_t> &filter) {
    for (int y = 1; y < in.height - 1; ++y) {
        for (int x = 1; x < in.width - 1; ++x) {
            int16_t sum = 0;

            for (int _x = -1; _x <= 1; ++_x) {
                for (int _y = -1; _y <= 1; ++_y) {
                    sum += (int16_t)filter(_x + 1, _y + 1) * (int16_t)in(x + _x, y + _y);
                }
            }

            if (sum > 255) {
                sum = 255;
            } else if (sum < 0) {
                sum = 0;
            }

            out(x, y) = sum;
        }
    }
}