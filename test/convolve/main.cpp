#include <iostream>
#include <chrono>
#include <thread>

#define WIDTH 160
#define HEIGHT 120

#include "map.h"
#include "color.h"
#include "kernels.cuh"
#include "camera.h"
#include "stereobm.h"
#include "image.h"

using namespace lm;
using namespace lm::autopilot;

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

int main() {

    map<grayscale> image(WIDTH, HEIGHT), result(WIDTH, HEIGHT);
    map<float> core(3, 3);

    core(0, 0) = -1;
    core(1, 0) = -1;
    core(2, 0) = -1;

    core(0, 1) = -1;
    core(1, 1) =  9;
    core(2, 1) = -1;

    core(0, 2) = -1;
    core(1, 2) = -1;
    core(2, 2) = -1;

    Camera camera(0, WIDTH, HEIGHT);

    map<grayscale, cuda_allocator<grayscale>> _image(WIDTH, HEIGHT), _result(WIDTH, HEIGHT);
    map<float, cuda_allocator<float>> _core(3, 3);

    auto p_image  = cuda_allocator<map<grayscale, cuda_allocator<grayscale>>>::allocate(1),
         p_result = cuda_allocator<map<grayscale, cuda_allocator<grayscale>>>::allocate(1);
                                      
    auto p_core = cuda_allocator<map<float, cuda_allocator<float>>>::allocate(1);

    cudaMemcpy(p_image, &_image, sizeof(_image), H2D);
    cudaMemcpy(p_core, &_core, sizeof(_core), H2D);
    cudaMemcpy(p_result, &_result, sizeof(_result), H2D);

    cudaMemcpy(_core.data(), core.data(), core.size() * sizeof(float), H2D);

    dim3 blocks(WIDTH / 8 + 1, HEIGHT / 8 + 1), threads(8, 8);
    void* convolve_args[] = {
        &p_image,
        &p_core,
        &p_result
    };

    while (1) {
        camera >> image;

        cudaMemcpy(_image.data(), image.data(), image.size(), H2D);

        cudaLaunchKernel((void*)convolve, blocks, threads, (void**)&convolve_args);
        cudaDeviceSynchronize();

        cudaMemcpy(result.data(), _result.data(), image.size(), D2H);
        
        Image img(result);
        img.save("image.png");

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}