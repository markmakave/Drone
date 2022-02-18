#include <cuda_runtime.h>
#include <stdio.h>
#include "pipeline.h"

__global__ void kernel(int *x, int *y) {
    printf("%d %d\n", *x, *y);
}

int main() {
    cudaDeviceReset();

    int x = 5;
    int *d_x;
    cudaMalloc((void**)&d_x, sizeof(x));
    cudaMemcpy(d_x, &x, sizeof(x), cudaMemcpyHostToDevice);

    int y = 6;
    int *d_y;
    cudaMalloc((void**)&d_y, sizeof(y));
    cudaMemcpy(d_y, &y, sizeof(y), cudaMemcpyHostToDevice);

    printf("Host: %p %p\n", d_x, d_y);
    
    Stage stage((void*)kernel, 2);
    stage.launch(1, 1, (void*)d_x, (void*)d_y);
    
    auto err = cudaGetLastError();
    if (err) {
        puts(cudaGetErrorString(err));
    }

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}