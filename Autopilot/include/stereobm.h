#pragma once
#include <cstdint>

#include <cuda_runtime.h>

#include "map.h"
#include "color.h"

namespace lm {

namespace autopilot {

//
// Standart CUDA allocator class
// uses cudaMalloc and cudaFree
//
template <typename Type>
class cuda_allocator {
    
    cuda_allocator() {};

public:

    // Allocates size enements of Type on device global memory
    static Type * allocate(size_t size) {
        if (size == 0) return nullptr;
        Type * ptr;
        cudaMalloc((void**)&ptr, size * sizeof(*ptr));
        return ptr;
    }

    // Deallocates memory followed by ptr
    static void deallocate(Type * ptr, size_t size) {
        cudaFree(ptr);
    }
};

// Stereo block matcher class
// contains matcher settings and frame maps
class StereoBM {

    // Block matcher block size
    int block_size;

    // Thresolds
    int validation_thresold;
    int distinction_threshold;

    // Camera setup parameters
    float focal_length;
    float camera_distance;

    // Map objects with device data pointers
    map<uint8_t, cuda_allocator<uint8_t>> cuda_left_frame, *cuda_left_frame_devptr,
                                          cuda_right_frame, *cuda_right_frame_devptr;
    map<int, cuda_allocator<int>> cuda_disparity_map, *cuda_disparity_map_devptr;
    map<float, cuda_allocator<float>> cuda_depth_map, *cuda_depth_map_devptr;
    
public:

    // Default constructor
    StereoBM(float focal_length,
             float distance,
             int block_size = 5,
             int distinction_threshold = 10,
             int validation_thresold = 10);

    // Computes disparity map
    void compute(const map<grayscale>& left_frame,
                 const map<grayscale>& right_frame,
                 map<float>& depth_map);

private:

    // Update device maps shapes and data pointers
    void _update(int width, int height);

};

}

}