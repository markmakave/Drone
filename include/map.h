#pragma once
#include <iostream>
#include <tuple>

namespace lm {

template <typename T>
class allocator {
        
    allocator() {};
    
public:

    static T* allocate(size_t size) {
        if (size == 0) return nullptr;
        return reinterpret_cast<T*>(operator new(size * sizeof(T)));
    }

    static void deallocate(T* ptr, size_t size) {
        operator delete(ptr);
    }
};

template <typename T, typename Allocator = allocator<T>>
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

    T& operator [] (size_t index) {
        return _data[index];
    }
    T operator [] (size_t index) const {
        return _data[index];
    }

    T& operator () (int x, int y) {
        return _data[y * _width + x];
    }
    T operator () (int x, int y) const {
        return _data[y * _width + x];
    }

    size_t size() const {
        return _width * _height;
    }

    int width() const {
        return _width;
    }

    int height() const {
        return _height;
    }

    T* data() const {
        return _data;
    }

    T& at(int x, int y) {
        if (x >= 0 && y >= 0 && x < _width && y < _height) {
            return this->operator()(x, y);
        }
        static T trash_bin = {0};
        return trash_bin;
    }

    T at(int x, int y) const {
        if (x >= 0 && y >= 0 && x < _width && y < _height) {
            return this->operator()(x, y);
        }
        return {0};
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
