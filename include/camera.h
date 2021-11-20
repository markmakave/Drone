#pragma once
#include <iostream>
#include <cstring>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

class Camera {
private:
    int         fd;
    uint8_t**   mem;

public:
            Camera();
            Camera(uint8_t);

    bool    info();

    bool    capture(uint8_t*);
    bool    start();
    bool    stop();

            ~Camera(); 

private:
    bool    _init_fmt();
    bool    _init_req();
    bool    _init_buf();   
};