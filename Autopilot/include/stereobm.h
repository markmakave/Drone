#pragma once
#include <cstdint>

#include <cuda_runtime.h>

#include "map.h"
#include "color.h"

namespace lm {

class StereoBM {

    int block_size;

    int validation_threshold;
    int distinction_threshold;

    float focal_length;
    float camera_distance;

public:

    StereoBM(float focal_length,
             float distance,
             int block_size = 5,
             int distinction_threshold = 10,
             int validation_threshold = 10);

    void compute(const map<grayscale>& left_frame,
                 const map<grayscale>& right_frame,
                 map<float>& depth_map);

};

namespace cuda {

template <typename Type>
class cuda_allocator {
    
    cuda_allocator() {};

public:

    static Type * allocate(size_t size) {
        if (size == 0) return nullptr;
        Type * ptr;
        cudaMalloc((void**)&ptr, size * sizeof(*ptr));
        return ptr;
    }

    static void deallocate(Type * ptr, size_t size) {
        cudaFree(ptr);
    }
};

class StereoBM {

    int block_size;

    int validation_threshold;
    int distinction_threshold;

    float focal_length;
    float camera_distance;

    map<uint8_t, cuda_allocator<uint8_t>> cuda_left_frame, *cuda_left_frame_devptr,
                                          cuda_right_frame, *cuda_right_frame_devptr;
    map<int, cuda_allocator<int>> cuda_disparity_map, *cuda_disparity_map_devptr;
    map<float, cuda_allocator<float>> cuda_depth_map, *cuda_depth_map_devptr;
    
public:

    StereoBM(float focal_length,
             float distance,
             int block_size = 5,
             int distinction_threshold = 10,
             int validation_threshold = 10);

    void compute(const map<grayscale>& left_frame,
                 const map<grayscale>& right_frame,
                 map<float>& depth_map);

private:

    void _update(int width, int height);

};

}

}