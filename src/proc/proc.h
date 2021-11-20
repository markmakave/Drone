#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "map.h"

void sobel(map<uint8_t> &, map<uint8_t> &, map<int8_t> &);