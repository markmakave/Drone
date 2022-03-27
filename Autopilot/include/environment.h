#include <vector>
#include <array>
#include <fstream>
#include <string>
#include <cstdint>

#include "map.h"
#include "geometry.h"

namespace lumina {

    class Environment {

        std::vector<dim> dots;

    public:

        Environment() {
        }

        void add(const dim& v) {
            dots.push_back(v);
        }

        void export_stl(const std::string& filename = "object.stl") {
            
            std::ofstream file(filename);

            if (!file) {
                return;
            }

            const char header[80] = { 0 };
            file.write(header, sizeof(header));
            uint32_t ntriangles = dots.size() * 12;
            file.write(reinterpret_cast<char*>(&ntriangles), sizeof(ntriangles));

            for (auto& dot : dots) {

                std::array<triangle, 4> triangles = {
                    triangle(dot + dim(0.0, 0.0,  1.0), dot + dim(-0.5, -0.5, -0.5), dot + dim( 0.5, -0.5, -0.5)),
                    triangle(dot + dim(0.0, 0.0,  1.0), dot + dim( 0.0,  0.5, -0.5), dot + dim(-0.5, -0.5, -0.5)),
                    triangle(dot + dim(0.0, 0.0,  1.0), dot + dim( 0.5, -0.5, -0.5), dot + dim( 0.0,  0.5, -0.5)),
                    triangle(dot + dim(0.0, 0.5, -0.5), dot + dim( 0.5, -0.5, -0.5), dot + dim(-0.5, -0.5, -0.5))
                };
                
                for (auto& trg : triangles) {
                    dim normal = trg.normal();
                    file.write(reinterpret_cast<char*>(&normal), sizeof(normal));
                    file.write(reinterpret_cast<char*>(&trg.v1), sizeof(trg.v1));
                    file.write(reinterpret_cast<char*>(&trg.v2), sizeof(trg.v2));
                    file.write(reinterpret_cast<char*>(&trg.v3), sizeof(trg.v3));
                    const char skip[2] = { 0 };
                    file.write(skip, sizeof(skip));
                }

            }

            file.close();

        }

    };

}