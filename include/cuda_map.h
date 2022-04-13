#pragma once

#include <cuda_runtime.h>
#include "map.h"

namespace lm {

namespace cuda {

enum TRANSFER_DIRECTION {
    H2H = cudaMemcpyHostToHost,
    H2D = cudaMemcpyHostToDevice,
    D2H = cudaMemcpyDeviceToHost,
    D2D = cudaMemcpyDeviceToDevice
};

template <typename T>
class allocator {
    
    allocator() = delete;

public:

    __host__ static T * allocate(size_t size = 1) {
        if (size == 0) return nullptr;
        T * ptr;
        cudaMalloc((void**)&ptr, size * sizeof(*ptr));
        return ptr;
    }

    __host__ static void deallocate(T * ptr, size_t size = 1) {
        cudaFree(ptr);
    }
};

template <typename T, typename Allocator = allocator<T>>
class map {
protected:

    int _width, _height;
    T *_data;
    map *_devptr;

public:

    __host__ map(int width = 0, int height = 0)
        : _width(width), _height(height), _devptr(nullptr) {
        _data = Allocator::allocate(this->size());
        this->_update();
    }

    __host__ map(const map& m)
        : map(m.width(), m.height()) {
        cudaMemcpy(_data, m.data(), this->size() * sizeof(T), D2D);
        this->_update();
    }

    __host__ map(map&& m) 
        : map(m.width(), m.height()) {
        _data = m._data;
        m._data = nullptr;
        this->_update();
    }

    __host__ ~map() {
        Allocator::deallocate(_data, this->size());
        allocator<map<T>>::deallocate(_devptr);
    }

    __host__ map& operator = (const map& m) {
        if (&m != this) {
            Allocator::deallocate(_data, size());
            _width = m._width;
            _height = m._height;
            _data = Allocator::allocate(size());
            cudaMemcpy(_data, m.data(), this->size() * sizeof(T), D2D);
            this->_update();
        }
        return *this;
    }

    __host__ map& operator = (map&& m) {
        if (&m != this) {
            Allocator::deallocate(_data, size());
            _width = m._width;
            _height = m._height;
            _data = m._data;
            m._data = nullptr;
            this->_update();
        }
        return *this;
    }

    __device__ T& operator [] (size_t index) {
        return _data[index];
    }
    __device__ T operator [] (size_t index) const {
        return _data[index];
    }

    __device__ T& operator () (int x, int y) {
        return _data[y * _width + x];
    }
    __device__ T operator () (int x, int y) const {
        return _data[y * _width + x];
    }

    __host__ __device__ size_t size() const {
        return _width * _height;
    }

    __host__ __device__ int width() const {
        return _width;
    }

    __host__ __device__ int height() const {
        return _height;
    }

    __host__ __device__ T* data() const {
        return _data;
    }

    __device__ T& at(int x, int y) {
        if (x >= 0 && y >= 0 && x < _width && y < _height) {
            return this->operator()(x, y);
        }
        static T trash_bin = {0};
        return trash_bin;
    }

    __device__ T at(int x, int y) const {
        if (x >= 0 && y >= 0 && x < _width && y < _height) {
            return this->operator()(x, y);
        }
        return {0};
    }

    __host__ void resize(int width, int height) {
        Allocator::deallocate(_data, this->size());
        _width = width;
        _height = height;
        _data = Allocator::allocate(size());
        this->_update();
    }

    // CUDA handle

    __host__ map* devptr() const {
        return _devptr;
    }

    __host__ void inject(const lm::map<T> &m) {
        cudaMemcpy(_data, m.data(), m.size() * sizeof(T), (cudaMemcpyKind)H2D);
    }

    __host__ void extract(lm::map<T> &m) {
        cudaMemcpy(m.data(), _data, m.size() * sizeof(T), (cudaMemcpyKind)D2H);
    }

private:

    __host__ void _update() {
        if (_devptr == nullptr) {
            _devptr = allocator<map<T>>::allocate();
        }
        cudaMemcpy(_devptr, this, sizeof(*this), (cudaMemcpyKind)H2D);
    }

};

}

}
