#include <iostream>
#include <cstdint>

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char** argv) {

	void* ptr = mmap(0, 100, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);

	return 0;
}
