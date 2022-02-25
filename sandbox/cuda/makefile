all:
	nvcc kernel.cu -c
	g++ main.cpp kernel.o -o main -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart