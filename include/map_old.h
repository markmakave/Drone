#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

enum TRANSFER_TYPE { 
    D2H, H2D
};

enum TYPE {
    NONE, DIVIDED, UNIFIED, USRPTR, DEVICE, HOST
};
template <class T>
class map {
public:
    uint16_t    width, height;
    T 	        *host_data, *dev_data;
    map<T>   	*dev_ptr;
    TYPE        type;

public:
    __host__            inline      map();
    __host__            inline      map(uint16_t, uint16_t, TYPE = DIVIDED, void* = nullptr);
    __host__            inline      map(const map&);
    __host__            inline      map(map&&) noexcept;

    __host__ __device__ inline T&   operator() (uint16_t, uint16_t);
    __host__ __device__ inline T&   operator[] (uint32_t);
    __host__            inline map& operator= (const map&);
    __host__            inline map& operator= (map&&) noexcept;

    __host__            inline void alloc();
    __host__            inline void transfer(TRANSFER_TYPE);

    __host__            inline      ~map();
};

template <class T>
map<T>::map() {
    width           = 0;
    height          = 0;
    host_data       = nullptr;
    type            = NONE;
    dev_ptr         = nullptr;
    dev_data        = nullptr;
}

template <class T>
map<T>::map(uint16_t width, uint16_t height, TYPE type, void* ptr) {
    this->width     = width;
    this->height    = height;
    this->type      = type;

    switch (type) {
        case DIVIDED:
            host_data       = new T[(size_t)width * height];
            dev_ptr         = nullptr;
            dev_data        = nullptr;
            break;

        case UNIFIED:
            cudaMallocManaged(&host_data, width * height * sizeof(T));
            dev_data        = host_data;
            cudaMalloc((void**)&dev_ptr, sizeof(*this));
            cudaMemcpy(dev_ptr, this, sizeof(*this), cudaMemcpyHostToDevice);
            break;

        case USRPTR:
            host_data       = (T*)ptr;
            dev_ptr         = nullptr;
            dev_data        = nullptr;
            break;

        case DEVICE:
            host_data       = nullptr;
            cudaMalloc((void**)&dev_ptr, sizeof(*this));
            cudaMalloc((void**)&dev_data, sizeof(T) * width * height);
            cudaMemcpy(dev_ptr, this, sizeof(*this), cudaMemcpyHostToDevice);
            break;

        case HOST:
            host_data       = new T[(size_t)width * height];
            dev_ptr         = nullptr;
            dev_data        = nullptr;
            break;
    }
}

template <class T>
map<T>::map(const map<T>& m) {
    width           = m.width;
    height          = m.height;
    
    size_t size     = (size_t)width * height;
    host_data       = new T[size];
    std::memcpy(host_data, m.host_data, size);

    dev_ptr         = nullptr;
    dev_data        = nullptr;
}

template <class T>
map<T>::map(map<T>&& m) noexcept {
    width           = m.width;
    height          = m.height;

    host_data       = m.host_data;
    m.host_data     = nullptr;

    dev_ptr         = m.dev_ptr;
    m.dev_ptr       = nullptr;

    dev_data        = m.dev_data;
    m.dev_data      = nullptr;
}

#ifdef __CUDA_ARCH__

template <class T>
T& map<T>::operator() (uint16_t x, uint16_t y) {
    return dev_data[y * width + x];
}

template <class T>
T& map<T>::operator[] (uint32_t i) {
    return dev_data[i];
}

#else

template <class T>
T& map<T>::operator() (uint16_t x, uint16_t y) {
    return host_data[y * width + x];
}

template <class T>
T& map<T>::operator[] (uint32_t i) {
    return host_data[i];
}

#endif

template <class T>
map<T>& map<T>::operator= (const map<T>& m) {
    if (&m == this) {
        return *this;
    }

    width           = m.width;
    height          = m.height;
    
    if (host_data) {
        delete[] host_data;
    }
    if (dev_ptr) {
        cudaFree(dev_ptr);
    }
    if (dev_data) {
        cudaFree(dev_data);
    }

    uint32_t size   = width * height;
    host_data       = new T[size];
    for (uint32_t i = 0; i < size; ++i) {
        host_data[i] = m.host_data[i];
    }

    dev_ptr         = nullptr;
    dev_data        = nullptr;

    return *this;
}

template <class T>
map<T>& map<T>::operator= (map<T>&& m) noexcept {
    if (&m == this) {
        return *this;
    }

    if (host_data) {
        delete[] host_data;
    }
    if (dev_ptr) {
        cudaFree(dev_ptr);
    }
    if (dev_data) {
        cudaFree(dev_data);
    }

    width           = m.width;
    height          = m.height;

    host_data       = m.host_data;
    m.host_data     = nullptr;

    dev_ptr         = m.dev_ptr;
    m.dev_ptr       = nullptr;

    dev_data        = m.dev_data;
    m.dev_data      = nullptr;

    return *this;
}

template <class T>
void map<T>::alloc() {
    cudaMalloc((void**)&(dev_ptr), sizeof(map<T>));
    cudaMalloc((void**)&(dev_data), width * height * sizeof(T));
    cudaMemcpy(dev_ptr, this, sizeof(map<T>), cudaMemcpyHostToDevice);
}

template <class T>
void map<T>::transfer(TRANSFER_TYPE type) {
    switch(type) {
        case D2H:
            cudaMemcpy(host_data, dev_data, width * height * sizeof(T), cudaMemcpyDeviceToHost);
            break;
        case H2D:
            cudaMemcpy(dev_data, host_data, width * height * sizeof(T), cudaMemcpyHostToDevice);
            break;
        default:
            break;
    }
}

// Destructor
template <class T>
map<T>::~map() {
    switch (type) {
        case NONE:
            break;

        case DIVIDED:
            host_data   ? delete[] host_data        : (void)0;

        case UNIFIED:
        case USRPTR:
        case DEVICE:
            dev_data    ? (void)cudaFree(dev_data)  : (void)0;
            dev_ptr     ? (void)cudaFree(dev_ptr)   : (void)0;
            break;

        default:
            std::cerr << "Error: bad map memory type\n";
    }
}