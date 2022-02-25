#pragma once
#include <iostream>
#include <chrono>

class Timer {

    std::chrono::time_point<std::chrono::high_resolution_clock> begin_timepoint, end_timepoint;

public:

    Timer() {
        start();
    }

    ~Timer() {
        stop();
        std::cout << "Scope escaped in " << elapsed() << "ms" << std::endl;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start() {
        begin_timepoint = std::chrono::high_resolution_clock::now();
        return begin_timepoint;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> stop() {
        end_timepoint = std::chrono::high_resolution_clock::now();
        return end_timepoint;
    }

    double elapsed() {
        auto begin_time = std::chrono::time_point_cast<std::chrono::microseconds>(begin_timepoint).time_since_epoch().count();
        auto end_time = std::chrono::time_point_cast<std::chrono::microseconds>(end_timepoint).time_since_epoch().count();
        return (end_time - begin_time) / 1000.0;
    }

};