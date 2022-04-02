#include <stdexcept>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "stereobm.h"
#include "kernels.cuh"
#include "map.h"
#include "color.h"

#define CUDA_BLOCK_WIDTH 8
#define CUDA_BLOCK_HEIGHT 8

using lm::map;

lm::autopilot::StereoBM::StereoBM(float focal_length,
                                  float camera_distance,
                                  int block_size,
                                  int distinction_threshold,
                                  int validation_thresold)
    : focal_length(focal_length),
      camera_distance(camera_distance),
      block_size(block_size),
      distinction_threshold(distinction_threshold),
      validation_thresold(validation_thresold)
{
    cuda_left_frame_devptr    = cuda_allocator<map<grayscale, cuda_allocator<grayscale>>>::allocate(1);
    cuda_right_frame_devptr   = cuda_allocator<map<grayscale, cuda_allocator<grayscale>>>::allocate(1);
    cuda_disparity_map_devptr = cuda_allocator<map<int, cuda_allocator<int>>>::allocate(1);
    cuda_depth_map_devptr     = cuda_allocator<map<float, cuda_allocator<float>>>  ::allocate(1);
}

void lm::autopilot::StereoBM::compute(const map<grayscale> & left_frame,
                                      const map<grayscale> & right_frame, 
                                      map<float> & result)
{
    // Input frames shapes mismatch check
    if (left_frame.width() != right_frame.width() || left_frame.height() != right_frame.height()) {
        throw std::runtime_error("Frame dimensions mismatch");
    }

    // Imput frames and internal frames shapes mismatch check
    if (left_frame.width() != cuda_left_frame.width() || left_frame.height() != cuda_left_frame.height()) {
        _update(left_frame.width(), left_frame.height());
    }

    // Output frame mismatch check
    if (left_frame.width() != result.width() || left_frame.height() != result.height()) {
        result.resize(left_frame.width(), left_frame.height());
    }
    
    // Copying raw frames data to the cude device
    cudaMemcpy(cuda_left_frame.data(),  left_frame.data(),  left_frame.size(),  cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_right_frame.data(), right_frame.data(), right_frame.size(), cudaMemcpyHostToDevice);

    // Internal cuda structures initialize
    dim3 blocks(cuda_depth_map.width() / CUDA_BLOCK_WIDTH + 1, 
                cuda_depth_map.height() / CUDA_BLOCK_HEIGHT + 1), 
         threads(CUDA_BLOCK_WIDTH, CUDA_BLOCK_HEIGHT);

    // Disparity kernel launch sequence
    int block_radius = block_size / 2;
    void* disparity_args[] = {
        &cuda_left_frame_devptr, 
        &cuda_right_frame_devptr,
        &cuda_disparity_map_devptr,
        &block_radius,
        &distinction_threshold,
        &validation_thresold
    };
    cudaLaunchKernel((void*)disparity, blocks, threads, (void**)&disparity_args, 0);

    cudaDeviceSynchronize();

    // Depth kernel launch sequence
    void* depth_args[] = {
        &cuda_disparity_map_devptr,
        &cuda_depth_map_devptr,
        &focal_length,
        &camera_distance
    };
    cudaLaunchKernel((void*)depth, blocks, threads, (void**)&depth_args, 0);

    // Taking the result
    cudaMemcpy(result.data(), cuda_depth_map.data(), cuda_depth_map.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

void lm::autopilot::StereoBM::_update(int width, int height) {

    cuda_left_frame   .resize(width, height);
    cuda_right_frame  .resize(width, height);
    cuda_disparity_map.resize(width, height);
    cuda_depth_map    .resize(width, height);

    cudaMemcpy(cuda_left_frame_devptr,    &cuda_left_frame,    sizeof(cuda_left_frame),    cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_right_frame_devptr,   &cuda_right_frame,   sizeof(cuda_right_frame),   cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_disparity_map_devptr, &cuda_disparity_map, sizeof(cuda_disparity_map), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_depth_map_devptr,     &cuda_depth_map,     sizeof(cuda_depth_map),     cudaMemcpyHostToDevice);
}
