#include <climits>

#include "kernels.cuh"
#include "map.h"

using lm::map;

__global__ void lm::cuda::disparity(const map<uint8_t>* left,
                                         const map<uint8_t>* right,
                                         map<int>* disparity,
                                         const int block_radius,
                                         const int distinction_threshold,
                                         const int validation_threshold)
{
    // Current pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Out of bounds check
    if (x < block_radius || 
        y < block_radius || 
        x >= disparity->width() - block_radius || 
        y >= disparity->height() - block_radius)
    {
        disparity->at(x, y) = -1;
        return;
    }

    // Initial infimum values
    int infimum_position = x;
    int infimum_value = INT_MAX;

    // Right frame epipolar line walkthrough
    for (int center = block_radius; center <= x; ++center) {
        int hamming_sum = 0;

        // Block walkthrough
        for (int x_offset = -block_radius; x_offset <= block_radius; ++x_offset) {
            for (int y_offset = -block_radius; y_offset <= block_radius; ++y_offset) {
                int cur_difference = (int)(*left)(x + x_offset, y + y_offset) - (int)(*right)(center + x_offset, y + y_offset);
                if (abs(cur_difference) > distinction_threshold)
                    hamming_sum++;
            }
        }

        // Selecting miminum hamming sum
        if (hamming_sum < infimum_value) {
            infimum_position = center;
            infimum_value = hamming_sum;
        }
    }

    // Validation
    if (infimum_value <= validation_threshold) {
        disparity->operator()(x, y) = x - infimum_position;
    } else {
        disparity->operator()(x, y) = -1;
    }
}

__global__ void lm::cuda::depth(const map<int>* disparity,
                                      map<float>* depth,
                                const float focal_length,
                                const float camera_distance)
{
    // Current pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Out of bounds check
    if (x >= depth->width()|| 
        y >= depth->height())
    {
        return;
    }

    // Validation
    int current = disparity->operator()(x, y);
    if (current <= 0) {
        depth->operator()(x, y) = -1.f;
    } else {
        depth->operator()(x, y) = focal_length * camera_distance / current;
    }
}

static __device__ lm::grayscale clamp(int x) {
    if (x > 255)
        return 255;
    if (x < 0)
        return 0;
    return x;
}

__global__ void lm::cuda::convolve(const map<grayscale> *input,
                                        const map<float> *core,
                                              map<grayscale> *output)
{
    // Current pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Out of bounds check
    if (x >= input->width() ||
        y >= input->height())
    {
        return;
    }

    int sum = 0;
    for (int i = 0; i < core->width(); ++i)
    {
        for (int j = 0; j < core->height(); ++j)
        {
            sum += input->at(x + i - core->width() / 2, y + j - core->height() / 2) * core->operator()(i, j);
        }
    }

    output->operator()(x, y) = ::clamp(sum);
}