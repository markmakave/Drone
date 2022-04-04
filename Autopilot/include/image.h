#pragma once

#include <fstream>
#include <string>
#include <map>
#include <functional>
#include <stdexcept>

#include <png++/png.hpp>

#include "color.h"
#include "map.h"

namespace lm {

class Image : public map<rgba> {
public:

    Image(int width = 0, int height = 0)
        : map<rgba>(width, height) {
    }

    Image(const map<rgba>& raw)
        : map<rgba>(raw) {
    }

    Image(map<rgba>&& raw)
        : map<rgba>(raw) {
    }

    Image& vflip() {
        
        return *this;
    }

    Image& hflip() {

        return *this;
    }

    void save(const std::string& filename) {

        static const std::map<std::string, std::function<void()>> ops = {
            {
                ".tga", [this, &filename](){
                    std::ofstream file(filename);
                    if (!file) return;

                    static char header[] = {
                        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
                        0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
                        0x20, 0b00101000
                    };

                    *(reinterpret_cast<uint16_t*>(&header[12])) = this->width();
                    *(reinterpret_cast<uint16_t*>(&header[14])) = this->height();
                    
                    file.write(header, sizeof(header));
                    file.write(reinterpret_cast<char*>(this->data()), this->size() * sizeof(rgba));
                    file.close();
                }
            },
            {
                ".png", [this, &filename](){
                    png::image<png::rgba_pixel> png_image(this->width(), this->height());
                    for (size_t y = 0; y < png_image.get_height(); ++y) {
                        for (size_t x = 0; x < png_image.get_width(); ++x) {
                            lm::rgba color = this->operator()(x, y);
                            png_image[y][x] = png::rgba_pixel(color.r, color.g, color.b, color.a);
                        }
                    }
                    png_image.write(filename);
                }
            }
        };

        size_t match = filename.rfind('.');
        if (match != std::string::npos) {
            std::string extension = filename.substr(match);
            std::map<std::string, std::function<void()>>::const_iterator op;
            if ((op = ops.find(extension)) != ops.end()) {
                op->second();
            } else {
                throw std::runtime_error("Invalid file extension");
            }
        } else {
            throw std::runtime_error("Invalid file name");
        }
    }
};

map<rgba> gradient(const map<float>& image, const float floor = 0.f, const float roof = 100.f) {
    map<rgba> grad(image.width(), image.height());

    for (int x = 0; x < image.width(); ++x) {
        for (int y = 0; y < image.height(); ++y) {
            float current = image(x, y);

            if (current < 0) {
                grad(x ,y) = rgba(0, 0, 0, 0);
                continue;
            }

            if (current <= floor) {
                grad(x ,y) = rgba(255, 0, 0);
            } else if (current >= roof) {
                grad(x, y) = rgba(0, 0, 255);
            } else {
                float factor = (current - floor) / (roof - floor);
                grad(x, y) = rgba(255 * (1.f - factor), 0,  255 * factor);
            }
        }
    }

    return grad;
}

}