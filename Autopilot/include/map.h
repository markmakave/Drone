#pragma once
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

namespace lm {

template <typename T>
class default_allocator {
        
    default_allocator() {};
    
public:

    static T* allocate(size_t size) {
        if (size == 0) return nullptr;
        return reinterpret_cast<T*>(operator new(size * sizeof(T)));
    }

    static void deallocate(T* ptr, size_t size) {
        operator delete(ptr);
    }
};

template <typename T, typename Allocator = default_allocator<T>>
class map {
protected:

    int _width, _height;
    T* _data;

public:

    map(int width = 0, int height = 0)
        : _width(width), _height(height) {
        _data = Allocator::allocate(size());
    }

    map(const map& m)
        : _width(m._width), _height(m._height) {
        _data = Allocator::allocate(size());
        for (size_t i = 0; i < size(); ++i) {
            _data[i] = m._data[i];
        }
    }

    map(map&& m) 
        : _width(m._width), _height(m._height) {
        _data = m._data;
        m._data = nullptr;
    }

    ~map() {
        Allocator::deallocate(_data, size());
    }

    map& operator = (const map& m) {
        if (&m != this) {
            Allocator::deallocate(_data, size());
            _width = m._width;
            _height = m._height;
            _data = Allocator::allocate(size());
            for (size_t i = 0; i < size(); ++i) {
                _data[i] = m._data[i];
            }
        }
        return *this;
    }

    map& operator = (map&& m) {
        if (&m != this) {
            Allocator::deallocate(_data, size());
            _width = m._width;
            _height = m._height;
            _data = m._data;
            m._data = nullptr;
        }
        return *this;
    }

    HOST DEVICE T& operator [] (size_t index) {
        return _data[index];
    }
    HOST DEVICE T operator [] (size_t index) const {
        return _data[index];
    }

    HOST DEVICE T& operator () (int x, int y) {
        return _data[y * _width + x];
    }
    HOST DEVICE T operator () (int x, int y) const {
        return _data[y * _width + x];
    }

    HOST DEVICE size_t size() const {
        return _width * _height;
    }

    HOST DEVICE int width() const {
        return _width;
    }

    HOST DEVICE int height() const {
        return _height;
    }

    HOST DEVICE T* data() const {
        return _data;
    }

    HOST DEVICE T* at(int x, int y) {
        if (x >= 0 && y >= 0 && x < _width && y < _height) {
            return this->operator()(x, y);
        }
        return T();
    }

    void resize(int width, int height) {
        Allocator::deallocate(_data, size());
        _width = width;
        _height = height;
        _data = Allocator::allocate(size());
    }

    friend std::ostream& operator << (std::ostream& out, const map& m) {
        for (size_t y = 0; y < m._height; ++y) {
            for (size_t x = 0; x < m._width; ++x) {
                out << m(x, y);
            }
            out << std::endl;
        }
        return out;
    }

};

}
