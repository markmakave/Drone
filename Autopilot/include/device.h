#include <iostream>
#include <stdexcept>

#include <unistd.h>
#include <fcntl.h>

namespace lm {

class Device {
protected:
    int fd;

public:

    Device(const std::string& filepath) {
        fd = open(filepath.c_str(), O_RDWR);
        if (fd < 0) {
            throw std::runtime_error("Device file open failed");
        }
    }

    virtual void info() {
        std::cout << "No device info provided" << std::endl;
    }

    ~Device() {
        close(fd);
    }

};

}