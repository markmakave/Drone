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

///////////////////////////////////////////////////////////////////////////////////////////////////

lm::StereoBM::StereoBM(
    float focal_length,
    float camera_distance,
    int   median_block_size,
    int   disparity_block_size,
    int   disparity_threshold)

    : focal_length(focal_length),
      camera_distance(camera_distance),
      median_block_size(median_block_size),
      disparity_block_size(disparity_block_size),
      disparity_threshold(disparity_threshold)
{ 
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void lm::StereoBM::compute(
    const map<grayscale> &left_frame,
    const map<grayscale> &right_frame,
          map<float>     &depth_map)
{
    // Input frames shapes mismatch check
    if (left_frame.width() != right_frame.width() || left_frame.height() != right_frame.height()) {
        throw std::runtime_error("Frame dimensions mismatch");
    }

    int width = left_frame.width(), height = left_frame.height();
    int block_radius = disparity_block_size / 2;

    if (depth_map.width() != width || depth_map.height() != height) {
        depth_map.resize(width, height);
    }

    #pragma omp parallel for
    for (int x = 0; x < width; ++x) {
        #pragma omp parallel for
        for (int y = 0; y < height; ++y) {

            if (x < block_radius || x >= width - block_radius ||
                y < block_radius || y >= height - block_radius) {
                depth_map(x, y) = -1.f;
                continue;
            }

            // Initial infimum values
            int infimum_position = x;
            int infimum_value = INT_MAX;

            // Right frame epipolar line walkthrough
            for (int center = block_radius; center <= x; ++center) {
                int difference = 0;

                // Block walkthrough
                for (int x_offset = -block_radius; x_offset <= block_radius; ++x_offset) {
                    for (int y_offset = -block_radius; y_offset <= block_radius; ++y_offset) {
                        int cur_difference = (int)left_frame(x + x_offset, y + y_offset) - (int)right_frame(center + x_offset, y + y_offset);
                        difference += abs(cur_difference);
                    }
                }

                // Selecting miminum hamming sum
                if (difference < infimum_value) {
                    infimum_position = center;
                    infimum_value = difference;
                }
            }

            if (infimum_value <= disparity_threshold) {
                depth_map(x, y) = focal_length * camera_distance / (x - infimum_position);
            } else {
                depth_map(x, y) = -1.f;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

lm::cuda::StereoBM::StereoBM(
    float focal_length,
    float camera_distance,
    int   median_block_size,
    int   disparity_block_size,
    int   disparity_threshold)

    : focal_length(focal_length),
      camera_distance(camera_distance),
      median_block_size(median_block_size),
      disparity_block_size(disparity_block_size),
      disparity_threshold(disparity_threshold)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void lm::cuda::StereoBM::compute(
    const lm::map<grayscale> &left_frame,
    const lm::map<grayscale> &right_frame, 
          lm::map<float>     &result)
{
    // Input frames shapes mismatch check
    if (left_frame.width() != right_frame.width() || left_frame.height() != right_frame.height()) {
        throw std::runtime_error("Frame dimensions mismatch");
    }

    int width = left_frame.width(), height = left_frame.height();

    // Imput frames and internal frames shapes mismatch check
    if (cuda_left_frame.width() != width || cuda_left_frame.height() != height) {
        this->_update(width, height);
    }

    // Output frame mismatch check
    if (result.width() != width || result.height() != height) {
        result.resize(width, height);
    }
    
    // Copying raw frames data to the cude device
    cuda_left_frame.inject(left_frame);
    cuda_right_frame.inject(right_frame);

    auto left_devptr         = cuda_left_frame.devptr();
    auto right_devptr        = cuda_right_frame.devptr();
    auto left_median_devptr  = cuda_left_median.devptr();
    auto right_median_devptr = cuda_right_median.devptr();
    auto disparity_devptr    = cuda_disparity.devptr();
    auto depth_devptr        = cuda_depth.devptr();

    // Internal cuda structures initialize
    dim3 blocks(width / CUDA_BLOCK_WIDTH + 1, height / CUDA_BLOCK_HEIGHT + 1), 
         threads(CUDA_BLOCK_WIDTH, CUDA_BLOCK_HEIGHT);

    // Median kernel launch sequence
    int median_block_radius = median_block_size / 2;

    void* left_median_args[] = {
        &left_devptr,
        &left_median_devptr,
        &median_block_radius
    };
    cudaLaunchKernel((void*)median, blocks, threads, (void**)&left_median_args, 0);

    void* right_median_args[] = {
        &right_devptr,
        &right_median_devptr,
        &median_block_radius
    };
    cudaLaunchKernel((void*)median, blocks, threads, (void**)&right_median_args, 0);

    // Disparity kernel launch sequence
    int disparity_block_radius = disparity_block_size / 2;

    void* disparity_args[] = {
        &left_median_devptr, 
        &right_median_devptr,
        &disparity_devptr,
        &disparity_block_radius,
        &disparity_threshold
    };
    cudaLaunchKernel((void*)disparity, blocks, threads, (void**)&disparity_args, 0);

    // Depth kernel launch sequence
    void* depth_args[] = {
        &disparity_devptr,
        &depth_devptr,
        &focal_length,
        &camera_distance
    };
    cudaLaunchKernel((void*)depth, blocks, threads, (void**)&depth_args, 0);
    
    cudaDeviceSynchronize();

    // Taking the result
    cuda_depth.extract(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void lm::cuda::StereoBM::_update(int width, int height) {
    cuda_left_frame  .resize(width, height);
    cuda_right_frame .resize(width, height);

    cuda_left_median .resize(width, height);
    cuda_right_median.resize(width, height);

    cuda_disparity   .resize(width, height);
    
    cuda_depth       .resize(width, height);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
