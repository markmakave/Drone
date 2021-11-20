#include <cuda_runtime.h>
#include <cstdint>

#include "map.h"

__global__ void depth(map<uint8_t>*, map<uint8_t>*, map<uint8_t>*);
__global__ void filter(map<uint8_t>*, map<uint8_t>*);