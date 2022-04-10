#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

#include <cstdint>

#include <unistd.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include "device.h"
#include "color.h"
#include "map.h"

namespace lm {

static int xioctl(int fd, int request, void *arg) {
    int r;
    do {
        r = ioctl (fd, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

class Camera : public Device {

    int id;
    int width, height;
    std::vector<uint8_t*> buffers;
    bool streaming;

public:
    
    Camera(int id, int width = 640, int height = 480, int buffer_count = 10) 
        : Device(std::string("/dev/video") + std::to_string(id)), width(width), height(height), streaming(false), id(id) {

        buffers.resize(buffer_count);

        _pass_format();
        _request_buffers();
        _allocate_buffers();
    }

    void info() override {
        v4l2_capability cap;
        if (xioctl(fd, VIDIOC_QUERYCAP, &cap) != 0) {
            throw std::runtime_error("Camera info request failed");
            return;
        }

        std::cout <<
            "Camera #" << id << "\n\t" <<
                "Driver:        " << cap.driver   << "\n\t" <<
                "Card:          " << cap.card     << "\n\t" <<
                "Bus info:      " << cap.bus_info << "\n\t" <<
                "Capabilities:  \n";

        
        if(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)
            std::cout << "\t\tv4l2 dev support capture\n";

        if(cap.capabilities & V4L2_CAP_VIDEO_OUTPUT)
            std::cout << "\t\tv4l2 dev support output\n";

        if(cap.capabilities & V4L2_CAP_VIDEO_OVERLAY)
            std::cout << "\t\tv4l2 dev support overlay\n";

        if(cap.capabilities & V4L2_CAP_STREAMING)
            std::cout << "\t\tv4l2 dev support streaming\n";

        if(cap.capabilities & V4L2_CAP_READWRITE)
            std::cout << "\t\tv4l2 dev support read write\n";
        
    }

    void start() {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd, VIDIOC_STREAMON, &type) != 0) {
            throw std::runtime_error("Camera stream starting failed");
        }

        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        for (buf.index = 0; buf.index < buffers.size(); ++buf.index) {
            if(xioctl(fd, VIDIOC_QBUF, &buf) != 0) {
                throw std::runtime_error("Camera queuing buffer failed");
            }
        }

        streaming = true;
    }

    void stop() {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd, VIDIOC_STREAMOFF, &type) != 0) {
            throw std::runtime_error("Camera stream stoping failed");
        }
        streaming = false;
    }

    Camera& operator >> (map<grayscale>& frame) {
        if (width != frame.width() || height != frame.height()) {
            frame.resize(width, height);
        }
        
        if (!streaming) {
            start();
        }

        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if(xioctl(fd, VIDIOC_DQBUF, &buf) != 0) {
            throw std::runtime_error("Camera buffer dequeuing failed");
        }

        for (size_t i = 0; i < frame.size(); ++i) {
            frame[i] = buffers[buf.index][i * 2];
        }

        if(xioctl(fd, VIDIOC_QBUF, &buf) != 0) {
            throw std::runtime_error("Camera buffer queuing failed");
        }

        return *this;
    }

    Camera& operator >> (map<rgba>& frame) {
        if (width != frame.width() || height != frame.height()) {
            frame.resize(width, height);
        }
        
        if (!streaming) {
            start();
        }

        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if(xioctl(fd, VIDIOC_DQBUF, &buf) != 0) {
            throw std::runtime_error("Camera buffer dequeuing failed");
        }

        for (size_t pixel = 0, offset = 0; pixel < frame.size(); pixel += 2, offset += 4) {
            uint8_t y1  = buffers[buf.index][offset + 0];
            uint8_t u   = buffers[buf.index][offset + 1];
            uint8_t y2  = buffers[buf.index][offset + 2];
            uint8_t v   = buffers[buf.index][offset + 3];

            frame[pixel]        = rgba(yuyv(y1, u, v));
            frame[pixel + 1]    = rgba(yuyv(y2, u, v));
        }

        if(xioctl(fd, VIDIOC_QBUF, &buf) != 0) {
            throw std::runtime_error("Camera buffer queuing failed");
        }

        return *this;
    }

    ~Camera() {
        if (streaming) {
            stop();
        }
        for (auto buffer : buffers) {
            munmap(buffer, width * height * 2);
        }
    } 

private:

    void _pass_format() {;
        v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (xioctl(fd, VIDIOC_S_FMT, &fmt) != 0) {
            throw std::runtime_error("Camera format setting failed");
        }
    }

    void _request_buffers() {
        v4l2_requestbuffers req = {};
        req.count = buffers.size();
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (xioctl(fd, VIDIOC_REQBUFS, &req) != 0) {
            throw std::runtime_error("Camera requesting buffers failed");
        }
    }

    void _allocate_buffers() {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        for (buf.index = 0; buf.index < buffers.size(); ++buf.index) {
            if (xioctl(fd, VIDIOC_QUERYBUF, &buf) != 0) {
                throw std::runtime_error("Camera quering buffer failed");
            }

            buffers[buf.index] = static_cast<uint8_t*>(mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset));
            if (buffers[buf.index] == MAP_FAILED) {
                throw std::runtime_error("Camera buffer mapping failed");
            }
        }
    }

};

}