#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(int x, int y) {
    printf("Hello from GPU! %d %d\n", x, y);
}