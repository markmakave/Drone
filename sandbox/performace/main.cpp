#include <iostream>
#include <thread>
#include <vector>

#include <unistd.h>
#include <fcntl.h>

void workload(std::vector<int> &x, int num, int &res) {
    size_t offset = x.size() / 4 * num;
    int max = x[offset];
    for (size_t i = 0; i < x.size() / 4; ++i) {
        if (x[i + offset] > max) {
            max = x[i + offset];
        }
    }
    res = max;
}

//#define ASYNC

int main() {

    std::vector<int> x(1000000000);
    int file = open("/dev/random", O_RDONLY);
    read(file, x.data(), x.size() * sizeof(int));
    close(file);

    int final;

    #ifndef ASYNC

    {
        final = x[0];
        #pragma omp parallel for
        for (size_t i = 0; i < x.size(); ++i) {
            if (x[i] > final) {
                #pragma omp critical
                final = x[i];
            }
        }
    }

    #else

    {

    std::vector<int> max(4);

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.push_back(std::thread(workload, std::ref(x), i, std::ref(max[i])));
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    final = max[0];
    for (auto temp : max) {
        if (temp > final) {
            final = temp;
        }
    }

    }

    #endif
    
    std::cout << final << std::endl;

    return 0;
}

