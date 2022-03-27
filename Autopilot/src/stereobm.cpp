#include <stdexcept>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "stereobm.h"
#include "kernels.h"
#include "map.h"

#define CUDA_BLOCK_WIDTH 8
#define CUDA_BLOCK_HEIGHT 8

using lumina::map;

lumina::StereoBM::StereoBM(int block_size, int thresold)
    : block_size(block_size), thresold(thresold) {
    cuda_left_frame_devptr   = cuda_allocator<map<uint8_t, cuda_allocator<uint8_t>>>::allocate(1);
    cuda_right_frame_devptr  = cuda_allocator<map<uint8_t, cuda_allocator<uint8_t>>>::allocate(1);
    cuda_left_median_devptr  = cuda_allocator<map<uint8_t, cuda_allocator<uint8_t>>>::allocate(1);
    cuda_right_median_devptr = cuda_allocator<map<uint8_t, cuda_allocator<uint8_t>>>::allocate(1);
    cuda_depth_map_devptr    = cuda_allocator<map<uint8_t, cuda_allocator<uint8_t>>>::allocate(1);
}

void lumina::StereoBM::compute(map<uint8_t> & left_frame, map<uint8_t> & right_frame, map<uint8_t> & result) {
    if (left_frame.width() != right_frame.width() || left_frame.height() != right_frame.height()) {
        throw std::runtime_error("Frame dimensions are not equal");
    }

    if (left_frame.width() != cuda_left_frame.width() || left_frame.height() != cuda_left_frame.height()) {
        _update(left_frame.width(), left_frame.height());
    }

    if (left_frame.width() != result.width() || left_frame.height() != result.height()) {
        result.resize(left_frame.width(), left_frame.height());
    }
    
    cudaMemcpy(cuda_left_frame.data(),  left_frame.data(),  left_frame.size(),  cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_right_frame.data(), right_frame.data(), right_frame.size(), cudaMemcpyHostToDevice);

    dim3 blocks(cuda_depth_map.width() / CUDA_BLOCK_WIDTH + 1, cuda_depth_map.height() / CUDA_BLOCK_HEIGHT + 1), threads(CUDA_BLOCK_WIDTH, CUDA_BLOCK_HEIGHT);

    void* left_median_args[] = { &cuda_left_frame_devptr, &cuda_left_median_devptr };
    cudaLaunchKernel((void*)median, blocks, threads, (void**)&left_median_args, 0);
    
    void* right_median_args[] = { &cuda_right_frame_devptr, &cuda_right_median_devptr };
    cudaLaunchKernel((void*)median, blocks, threads, (void**)&right_median_args, 0);

    cudaDeviceSynchronize();

    int radius = block_size / 2;
    void* disparity_args[] = { &cuda_left_median_devptr, &cuda_right_median_devptr, &cuda_depth_map_devptr, &radius, &thresold };
    cudaLaunchKernel((void*)depth, blocks, threads, (void**)&disparity_args, 0);
 
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), cuda_depth_map.data(), cuda_depth_map.size(), cudaMemcpyDeviceToHost);
}

void lumina::StereoBM::_update(int width, int height) {

    cuda_left_frame  .resize(width, height);
    cuda_right_frame .resize(width, height);
    cuda_left_median .resize(width, height);
    cuda_right_median.resize(width, height);
    cuda_depth_map   .resize(width, height);

    cudaMemcpy(cuda_left_frame_devptr,   &cuda_left_frame,   sizeof(cuda_left_frame),   cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_right_frame_devptr,  &cuda_right_frame,  sizeof(cuda_right_frame),  cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_left_median_devptr,  &cuda_left_median,  sizeof(cuda_left_median),  cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_right_median_devptr, &cuda_right_median, sizeof(cuda_right_median), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_depth_map_devptr,    &cuda_depth_map,    sizeof(cuda_depth_map),    cudaMemcpyHostToDevice);
}
