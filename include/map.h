#pragma once
#include <cuda_runtime.h>
#include <cstdint>

enum TRANSFER_TYPE { 
    D2H, H2D
};

enum PTR_TYPE {
    SYS_PTR, USR_PTR
};
template <class T>
class map {
public:
    uint16_t    width, height;
    T* 	        host_data, *dev_data;
    map<T>* 	dev_ptr;
    PTR_TYPE 	ptr_type;

public:
    __host__ __device__ inline      map();
    __host__ __device__ inline      map(uint16_t, uint16_t);
    __host__ __device__ inline      map(uint16_t, uint16_t, void*);
    __host__            inline      map(const map&);
    __host__            inline      map(map&&) noexcept;

    __host__ __device__ inline T&   operator() (const int32_t, const int32_t);
    __host__ __device__ inline T&   operator[] (const int32_t);
    __host__            inline map& operator= (const map&);
    __host__            inline map& operator= (map&&) noexcept;

    __host__            inline void alloc();
    __host__            inline void transfer(TRANSFER_TYPE);

    __host__            inline      ~map();
};

// Default constructor
template <class T>
map<T>::map() {
    width           = 0;
    height          = 0;
    host_data       = nullptr;
    dev_ptr         = nullptr;
    dev_data        = nullptr;
}
// Size based constructor
template <class T>
map<T>::map(uint16_t width, uint16_t height) {
    this->width     = width;
    this->height    = height;
    host_data       = new T[width * height];
    ptr_type        = SYS_PTR;
    dev_ptr         = nullptr;
    dev_data        = nullptr;
}
// Ptr based constructor
template <class T>
map<T>::map(uint16_t width, uint16_t height, void* ptr) {
    this->width     = width;
    this->height    = height;
    host_data       = (uint8_t*)ptr;
    ptr_type        = USR_PTR;
    dev_ptr         = nullptr;
    dev_data        = nullptr;
}

// Deep copy constructor
template <class T>
map<T>::map(const map<T>& m) {
    width           = m.width;
    height          = m.height;
    
    uint32_t size   = width * height;
    host_data       = new T[size];
    for (uint32_t i = 0; i < size; ++i) {
        host_data[i] = m.host_data[i];
    }

    dev_ptr         = nullptr;
    dev_data        = nullptr;
}
// Move constructor
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
T& map<T>::operator() (const int32_t x, const int32_t y) {
    return dev_data[y * width + x];
}

template <class T>
T& map<T>::operator[] (const int32_t i) {
    return dev_data[i];
}

#else

template <class T>
T& map<T>::operator() (const int32_t x, const int32_t y) {
    return host_data[y * width + x];
}

template <class T>
T& map<T>::operator[] (const int32_t i) {
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

// Destructor
template <class T>
map<T>::~map() {
    if (host_data && ptr_type == SYS_PTR){
        delete[] host_data;
    }
    if (dev_ptr) {
        cudaFree(dev_ptr);
    }
    if (dev_data) {
        cudaFree(dev_data);
    }
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