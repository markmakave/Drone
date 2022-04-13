#include <cstdio>

#include "cuda_map.h"
#include "device_launch_parameters.h"

__global__ void kernel(lm::cuda::map<int> *m) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= m->width() || y >= m->height()) return;

    printf("%d ", m->operator()(x, y));
}

int main() {

    {
        lm::map<int> hm(5, 5);

        for (size_t i = 0; i < hm.size(); ++i) {
            hm[i] = i;
        }

        lm::cuda::map<int> dm(5, 5);

        dm.inject(hm);

        dim3 threads(8, 8);
        dim3 blocks(dm.width() / threads.x + 1, dm.height() / threads.y + 1);

        kernel <<<blocks, threads>>> (dm.devptr());
        cudaDeviceSynchronize();
    }

    cudaDeviceReset();

    return 0;
}