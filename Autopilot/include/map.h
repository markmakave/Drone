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

namespace lumina {

    template <typename Type>
    class default_allocator {
            
        default_allocator() {};
        
    public:

        static Type* allocate(size_t size) {
            if (size == 0) return nullptr;
            return reinterpret_cast<Type*>(operator new(size * sizeof(Type)));
        }

        static void deallocate(Type* ptr, size_t size) {
            operator delete(ptr);
        }
    };

    template <typename Type, typename Allocator = default_allocator<Type>>
    class map {

        int _width, _height;
        Type* _data;

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

        HOST DEVICE Type& operator [] (size_t index) {
            return _data[index];
        }
        HOST DEVICE Type operator [] (size_t index) const {
            return _data[index];
        }

        HOST DEVICE Type& operator () (int x, int y) {
            return _data[y * _width + x];
        }
        HOST DEVICE Type operator () (int x, int y) const {
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

        HOST DEVICE Type* data() const {
            return _data;
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
