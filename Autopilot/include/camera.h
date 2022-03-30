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

namespace autopilot {

class Camera : public Device {

    int width, height;
    std::vector<uint8_t*> buffers;
    bool streaming;

public:
    
    Camera(int id, int width = 640, int height = 480, int buffer_count = 10) 
        : Device(std::string("/dev/video") + std::to_string(id)), width(width), height(height), streaming(false) {

        buffers.resize(buffer_count);

        _pass_format();
        _request_buffers();
        _allocate_buffers();
    }

    void start() {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd, VIDIOC_STREAMON, &type) != 0) {
            throw std::runtime_error("Camera stream starting failed");
        }

        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        for (buf.index = 0; buf.index < buffers.size(); ++buf.index) {
            if(ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
                throw std::runtime_error("Camera queuing buffer failed");
            }
        }

        streaming = true;
    }

    void stop() {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd, VIDIOC_STREAMOFF, &type) != 0) {
            throw std::runtime_error("Camera stream stoping failed");
        }
        streaming = false;
    }

    Camera& operator >> (map<uint8_t>& frame) {
        if (width != frame.width() || height != frame.height()) {
            frame.resize(width, height);
        }
        
        if (!streaming) {
            start();
        }

        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if(ioctl(fd, VIDIOC_DQBUF, &buf) != 0) {
            throw std::runtime_error("Camera buffer dequeuing failed");
        }

        for (size_t i = 0; i < frame.size(); ++i) {
            frame[i] = buffers[buf.index][i * 2];
        }

        if(ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
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
        
        if(ioctl(fd, VIDIOC_DQBUF, &buf) != 0) {
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

        if(ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
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

        if (ioctl(fd, VIDIOC_S_FMT, &fmt) != 0) {
            throw std::runtime_error("Camera format setting failed");
        }
    }

    void _request_buffers() {
        v4l2_requestbuffers req = {};
        req.count = buffers.size();
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_REQBUFS, &req) != 0) {
            throw std::runtime_error("Camera requesting buffers failed");
        }
    }

    void _allocate_buffers() {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        for (buf.index = 0; buf.index < buffers.size(); ++buf.index) {
            if (ioctl(fd, VIDIOC_QUERYBUF, &buf) != 0) {
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

}