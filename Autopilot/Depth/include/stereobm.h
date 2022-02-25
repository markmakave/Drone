#pragma once
#include <cstdint>

#include <cuda_runtime.h>

#include "map.h"

namespace lumina {

    //
    // Standart CUDA allocator class
    // uses cudaMalloc and cudaFree
    //
    template <typename Type>
    class cuda_allocator {
        
        cuda_allocator() {};

    public:

        //
        // Allocates size enements of Type on device global memory
        //
        static Type * allocate(size_t size) {
            if (size == 0) return nullptr;
            Type * ptr;
            cudaMalloc((void**)&ptr, size * sizeof(*ptr));
            return ptr;
        }

        //
        // Deallocates memory followed by ptr
        //
        static void deallocate(Type * ptr, size_t size) {
            cudaFree(ptr);
        }
    };

    //
    // Stereo block matcher class
    // contains matcher settings and frame maps
    //
    class StereoBM {

        //
        // Block matcher block size
        //
        int block_size;
        int thresold;

        //
        // Map objects with device data pointers
        //
        map<uint8_t, cuda_allocator<uint8_t>> cuda_left_frame, 
                                              cuda_right_frame,
                                              cuda_depth_map;

        //
        // Device pointers to map objects above
        //
        map<uint8_t, cuda_allocator<uint8_t>> * cuda_left_frame_devptr,
                                              * cuda_right_frame_devptr,
                                              * cuda_depth_map_devptr;
        
    public:

        //
        // Default constructor
        //
        StereoBM(int block_size = 5, int thresold = 0);

        //
        // Computes disparity map
        //
        void compute(map<uint8_t> & left_frame, map<uint8_t> & right_frame, map<uint8_t> & result);

    private:


        //
        // Update device maps shapes and data pointers
        //
        void _update(int width, int height);

    };

}