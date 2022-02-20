#include <iostream>

#include <unistd.h>
#include <fcntl.h>

class Device {
protected:
    int fd;

public:

    Device(const std::string& filepath) {
        fd = open(filepath.c_str(), O_RDWR);
        if (fd < 0) {
            throw "Device file open failed";
        }
    }

    ~Device() {
        close(fd);
    }

};