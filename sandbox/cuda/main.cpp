#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern __global__ void kernel(int, int);

int main() {

    int x = 5, y = 10;
    void *arr[] = {&x, &y};
    cudaLaunchKernel((void*)kernel, 1, 1, (void**)&(void* arr[] = {&x, &y}), 0, 0);
    cudaDeviceSynchronize();

    return 0;
}