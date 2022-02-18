#pragma once
#include <cuda_runtime.h>
#include <cstdint>

class Stage {
public:
    uint8_t size;
    void*   kernel;

public:
    Stage()
        : size(0), kernel(nullptr) {
    }
    Stage(void* kernel, uint8_t size)
        : size(size), kernel(kernel) {
    }

    void launch(dim3 blocks, dim3 threads, void* args, ...) {
        auto start = &args;
        printf("Inside: %p %p\n", start[0], start[1]);
        
        //cudaLaunchKernel(kernel, blocks, threads, argv, 0, NULL);
        //cudaDeviceSynchronize();
    }
};

class Pipeline {
public:
    uint8_t length;
    Stage*  stages;

public:
    Pipeline();
    Pipeline(uint16_t);



};