#pragma once

#include <fstream>
#include <string>
#include <cstdint>
#include <cmath>

#include "map.h"
#include "geometry.h"

namespace lm {

namespace autopilot {

class Environment {

size_t width, height;
float fov;
map<dim> grid;

public:

    Environment(size_t width, size_t height, float fov = 45.f)
        : width(width), height(height), fov(fov / 180.f * M_PI) {
        grid = map<dim>(width, height);

        dim stepX, stepY;
        dim corner;
        float tan = std::tan(this->fov / 2.f);

        stepX = dim(0, -1,  0) * (tan * 2 / width);
        stepY = dim(0,  0, -1) * (stepX.len() / width * height);

        corner = dim(1, 0, 0) - stepX * (width / 2.f) - stepY * (height / 2.f);

        for (size_t x = 0; x < width; ++x) {
            for (size_t y = 0; y < height; ++y) {
                grid(x, y) = (corner + (stepX * float(x)) + (stepY * float(y))).normal();
            }
        }
    }

    void apply(const map<float>& depth) {
        if (depth.size() != grid.size()) {
            throw std::runtime_error("Maps size mismatch");
        }

        for (size_t i = 0; i < grid.size(); ++i) {
            grid[i] *= depth[i];
        }
    }

    void export_stl(const std::string& filename = "object.stl") {
        
        std::ofstream file(filename);
        if (!file) return;

        const char header[84] = { 0 };
        file.write(header, sizeof(header));

        uint32_t ntriangles = 0;
        for (int x = 0; x < grid.width() - 1; ++x) {
            for (int y = 0; y < grid.height() - 1; ++y) {
                #define EPS 1000
                if (grid(x, y).len() < EPS && grid(x + 1, y + 1).len() < EPS) {
                    
                    dim normal;
                    const char skip[2] = { 0 };

                    if (grid(x + 1, y).len() < EPS)
                    {
                        triangle trg(grid(x, y), grid(x + 1, y), grid(x + 1, y + 1));
                        file.write(reinterpret_cast<char*>(&normal), sizeof(normal));
                        file.write(reinterpret_cast<char*>(&trg), sizeof(trg));
                        file.write(skip, sizeof(skip));
                        ntriangles++;
                    }

                    if (grid(x, y + 1).len() < EPS)
                    {
                        triangle trg(grid(x, y), grid(x, y + 1), grid(x + 1, y + 1));
                        file.write(reinterpret_cast<char*>(&normal), sizeof(normal));
                        file.write(reinterpret_cast<char*>(&trg), sizeof(trg));
                        file.write(skip, sizeof(skip));
                        ntriangles++;
                    }
                }
            }
        }

        file.seekp(80);
        file.write(reinterpret_cast<char*>(&ntriangles), sizeof(ntriangles));

        file.close();

    }

};

}

}