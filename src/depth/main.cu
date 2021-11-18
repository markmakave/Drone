#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <unistd.h>
#include <fcntl.h>

#include "color.h"
#include "map.h"
#include "depth.h"

#define CAM_COUNT 6
#define FRAME_COUNT 100

const char* dataset[] = {
	"Aloe",
	//"self",
	"Bowling",
	"Cloth1",
	"Cloth2",
	"Cloth3",
	"Cloth4",
	"Flowerpots",
	"Lampshade",
	"Midd",
	"Monopoly",
	"Plastic",
	"Wood"
};

const uint8_t tga_header[] = { 
	0x00, 0x00, 0x03, 0x00, 
	0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00
};

map<rgba> read_tga(const char* filename) {
	int file = open(filename, O_RDONLY);
	if (file < 0) {
		std::cerr << "Error: no such file\n";
		return map<rgba>();
	}

	lseek(file, sizeof(tga_header), SEEK_SET);

	uint16_t width, height;
	read(file, &width, 2);
	read(file, &height, 2);
	map<rgba> result(width, height);

	lseek(file, 2, SEEK_CUR);

	read(file, result.host_data, width * height * sizeof(rgba));

	close(file);
	return result;
}

void write_tga(map<uint8_t> &m, const char* filename) {
	int file = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0666);
	
	write(file, tga_header, sizeof(tga_header));
	write(file, &m.width, 2);
	write(file, &m.height, 2);
	uint8_t bitness = 8;
	write(file, &bitness, 1);
	uint8_t type = 8;
	write(file, &type, 1);
	write(file, m.host_data, m.width * m.height * 2);

	close(file);
}

void mh(map<rgba> &in, map<uint8_t> &out) {
	for (uint32_t i = 0; i < in.width * in.height; ++i) {
		out.host_data[i] = ((uint16_t)(in.host_data[i].r) + in.host_data[i].g + in.host_data[i].b) / 3;
	}
}

int main() {
	cudaDeviceReset();

	// Init
	int width = 160, height = 120;

	map<rgba> left[CAM_COUNT], right[CAM_COUNT];
	map<uint8_t> depth_map[CAM_COUNT];

	char path[100];
	for (int i = 0; i < CAM_COUNT; ++i) {
		snprintf(path, sizeof(path), "../dataset/%s/left.tga", dataset[i]);
		left[i] = read_tga(path);

		snprintf(path, sizeof(path), "../dataset/%s/right.tga", dataset[i]);
		right[i] = read_tga(path);
	}

	for (int i = 0; i < CAM_COUNT; ++i) {
		depth_map[i] = map<uint8_t>(width, height);
	}
	
	map<uint8_t> mh_left[CAM_COUNT], mh_right[CAM_COUNT];
	for (int i = 0; i < CAM_COUNT; ++i) {
		mh_left[i] = map<uint8_t>(width, height);
		mh_right[i] = map<uint8_t>(width, height);
	}

	// Allocating mh and depth maps
	for (int i = 0; i < CAM_COUNT; ++i) {
		mh_left[i].alloc();
		mh_right[i].alloc();
	}

	for (int i = 0; i < CAM_COUNT; ++i) {
		depth_map[i].alloc();
	}

	// Cuda kernel config
	int tx = 16, ty = 16;
	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);

	auto start = std::chrono::high_resolution_clock::now();

	// FRAME
	for (int frame = 0; frame < FRAME_COUNT; ++frame)
	{
		// Making mh
		for(int i = 0; i < CAM_COUNT; ++i) {
			mh(left[i], mh_left[i]);
			mh(right[i], mh_right[i]);
		}

		// Transfering mh
		for (int i = 0; i < CAM_COUNT; ++i) {
			mh_left[i].transfer(H2D);
			mh_right[i].transfer(H2D);
		}

		// Kernel
		for (int cam = 0; cam < CAM_COUNT; ++cam) {
			depth <<<blocks, threads>>> (mh_left[cam].dev_ptr, mh_right[cam].dev_ptr, depth_map[cam].dev_ptr);
		}
		cudaDeviceSynchronize();

		//puts(cudaGetErrorString(cudaGetLastError()));

		// Transfering result
		for (int i = 0; i < CAM_COUNT; ++i) {
			depth_map[i].transfer(D2H);
		}

	}

	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

	char filename[100];
	for (int i = 0; i < CAM_COUNT; ++i) {
		snprintf(filename, sizeof(filename), "./result/%s_%03d.tga", "depth", i);
		write_tga(depth_map[i], filename);
	}

	std::cout << "Time elapsed: " << microseconds / 1000000.f << " seconds\n";
	std::cout << "Frame count: " << FRAME_COUNT << '\n';
	std::cout << "Avarage framerate: " << FRAME_COUNT / ((double)microseconds / (1000000)) << " frames per sec" << std::endl;

	return 0;
}
