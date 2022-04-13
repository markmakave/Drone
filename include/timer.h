#include <iostream>
#include <chrono>

class Timer {

    std::chrono::system_clock::time_point begin;

public:

    Timer() {
        begin = std::chrono::system_clock::now();
    }

    ~Timer() {
        auto end = std::chrono::system_clock::now();
        std::cout << "Elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << '\r';
        std::cout.flush();
    }

};