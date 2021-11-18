#include "camera.h"

enum { WIDTH = 160, HEIGHT = 120 };
#define BUFFER_COUNT 10
#define BUFFER_LENGTH WIDTH * HEIGHT * 2

Camera::Camera() {
    fd = -1;
}

Camera::Camera(uint8_t id) {
    char path[14];
    snprintf(path, sizeof(path), "/dev/video%d", id);

    fd = open(path, O_RDWR);
    if (fd < 0) {
        perror("Camera open");
        return;
    }

    mem = new uint8_t*[BUFFER_COUNT];
    _init_fmt();
    _init_req();
    _init_buf();
}

bool Camera::info() {
    v4l2_capability cap;
    if(ioctl(fd, VIDIOC_QUERYCAP, &cap) != 0) {
        perror("VIDIOC_QUERYCAP");
        return false;
    }

    printf("driver : %s\n",cap.driver);
    printf("card : %s\n",cap.card);
    printf("bus_info : %s\n",cap.bus_info);
    printf("version : %d.%d.%d\n",
        ((cap.version >> 16) & 0xFF),
        ((cap.version >> 8) & 0xFF),
        (cap.version & 0xFF));
    printf("capabilities: 0x%08x\n", cap.capabilities);
    printf("device capabilities: 0x%08x\n", cap.device_caps);

    return true;
}

bool Camera::capture(uint8_t* ptr) {
    v4l2_buffer buf = {};

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if(ioctl(fd, VIDIOC_DQBUF, &buf) != 0) {
        perror("VIDIOC_DQBUF");
        return false;
    }

    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        ptr[i] = mem[buf.index][i*2];
    }

    if(ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
        perror("VIDIOC_QBUF");
        return false;
    }
    return true;
}

bool Camera::start() {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) != 0) {
        perror("VIDIOC_STREAMON");
        return false;
    }
    return true;
}

bool Camera::stop() {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) != 0) {
        perror("VIDIOC_STREAMOFF");
        return false;
    }
    return true;
}

bool Camera::_init_fmt() {
    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) != 0) {
        perror("VIDIOC_S_FMT");
        return false;
    }

    return true;
}

bool Camera::_init_req() {
    v4l2_requestbuffers req = {};
    req.count = BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) != 0) {
        perror("VIDIOC_REQBUFS");
        return false;
    }

    return true;
}

bool Camera::_init_buf() {
    v4l2_buffer buf = {};
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) != 0) {
            perror("VIDIOC_QUERYBUF");
            return false;
        }

        mem[i] = (uint8_t*)mmap(NULL, BUFFER_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (mem[i] == MAP_FAILED) {
            perror("MMAP");
            return false;
        }
    }

    for (int i = 0; i < BUFFER_COUNT; ++i) {
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if(ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
            perror("VIDIOC_QBUF");
            return false;
        }
    }

    return true;
}

Camera::~Camera() {
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        munmap(mem[i], BUFFER_LENGTH);
    }
    delete[] mem;
    close(fd);
}