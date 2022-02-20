#pragma once
#include <iostream>

namespace lumina {

    template <typename T>
    class map {

        int width, height;
        T*  data;

    public:

        map(int width = 0, int height = 0)
            : width(width), height(height) {
            _allocate();
        }

        map(const map& m)
            : width(m.width), height(m.height) {
            _allocate();
            for (size_t i = 0; i < size(); ++i) {
                data[i] = m.data[i];
            }
        }

        map(map&& m) 
            : width(m.width), height(m.height) {
            data = m.data;
            m.data = nullptr;
        }

        ~map() {
            _deallocate();
        }

        map& operator = (const map& m) {
            if (&m == this) return *this;
            width = m.width;
            height = m.height;
            _deallocate();
            _allocate();
            for (size_t i = 0; i < size(); ++i) {
                data[i] = m.data[i];
            }
            return *this;
        }

        map& operator = (map&& m) {
            if (&m == this) return *this;;
            width = m.width;
            height = m.height;
            _deallocate();
            data = m.data;
            m.data = nullptr;
            return *this;
        }

        T& operator [] (size_t index) {
            return data[index];
        }
        T operator [] (size_t index) const {
            return data[index];
        }

        T& operator () (int x, int y) {
            return data[y * width + x];
        }
        T operator () (int x, int y) const {
            return data[y * width + x];
        }

        int size() const {
            return width * height;
        }

        int getw() const {
            return width;
        }
        int geth() const {
            return height;
        }

        friend std::ostream& operator << (std::ostream& out, const map& m) {
            for (size_t y = 0; y < m.height; ++y) {
                for (size_t x = 0; x < m.width; ++x) {
                    out << m(x, y);
                }
                out << std::endl;
            }
            return out;
        }

    private:

        void _allocate() {
            if (size()) {
                data = new T[size()];
            } else {
                data = nullptr;
            }
        }

        void _deallocate() {
            delete[] data;
        }

    };

}