#pragma once
#include <iostream>
#include <cstdint>
#include <cstring>  // for std::memcpy()

#ifdef __CUDACC__
#include <cuda_runtime.h>

/*
 *  Enumerates map memory types
 *  @param HOST host memory only
 *  @param DEVICE device memory only
 *  @param UNIFIER uses cuda unified memory system
*/
enum TYPE {
    NONE    = 0,
    HOST    = (1u << 0),
    DEVICE  = (1u << 1),
    UNIFIED = (1u << 2),
};
#define DEFAULT NONE

/*
 *  Enumerates transfer directions
 *  @param D2H Device to Host direction
 *  @param H2D Host to Device direction
*/
enum TRANSFER_TYPE { 
    D2H, H2D
};

/*
 *  Class containing a CPU/GPU locatable matrix
 *  @param width matrix width
 *  @param height matrix height
 *  @param *h_data pointer to host data
 *  @param *d_data pointer to device data
 *  @param *d_this pointer to device clone of the host object
*/
template <typename T> 
class map {
//private:
public:
    uint16_t    width, height;
    T           *h_data, *d_data;
    map         *d_this;
    uint8_t     type;

public:
    /*
     *  Default constructor
    */
    __host__ inline map()
        : width(0), height(0), h_data(nullptr), d_this(nullptr), d_data(nullptr), type(NONE) {
    }

    /*
     *  Constructor based on the matrix size
     *  @param width width of the matrix
     *  @param height height of the matrix
     *  @param type mep memory type
    */
    __host__ inline map(uint16_t width, uint16_t height, uint8_t type = DEFAULT)
        : width(width), height(height), type(type) {
        if (type) alloc(type);
    }

    /*
     *  Constructor based on other map object (deep copy contructor)
     *  @param m source matrix
    */
    __host__ inline map(const map &m)
        : width(m.width), height(m.height), type(m.type) {
        switch(type) {
            case NONE:
                h_data = nullptr;
                d_data = nullptr;
                d_this = nullptr;
                break;

            case HOST:
                _alloc_host();
                std::memcpy(h_data, m.h_data, size() * sizeof(T));
                d_data = nullptr;
                d_this = nullptr;
                break;

            case DEVICE:
                _alloc_device();
                cudaMemcpy(d_data, m.d_data, size() * sizeof(T), cudaMemcpyDeviceToDevice);
                h_data = nullptr;
                break;

            case HOST | DEVICE:
                _alloc_host();
                std::memcpy(h_data, m.h_data, size() * sizeof(T));
                _alloc_device();
                cudaMemcpy(d_data, m.d_data, size() * sizeof(T), cudaMemcpyDeviceToDevice);
                break;

            case UNIFIED:
                cudaMallocManaged((void**)&h_data, size() * sizeof(T));
                d_data = h_data;
                cudaMalloc((void**)&d_this, sizeof(*this));
                cudaMemcpy(d_this, this, sizeof(*this), cudaMemcpyHostToDevice);
                std::memcpy(h_data, m.h_data, size() * sizeof(T));
                break;

            default:
                std::cerr << "\033[31m" << "E: map at " << this << " - invalit memory type\n" << "\033[0m";
                break;
        }
    }

    /*
     *  Constructor based on other map object (move contructor)
     *  @param m source map object
    */
    __host__ inline map(map &&m) noexcept
        : width(m.width), height(m.height), h_data(m.h_data), d_data(m.d_data), d_this(m.d_this) {
        m.h_data    = nullptr;
        m.d_data    = nullptr;
        m.d_this    = nullptr;
        m.type      = NONE;
    }

    /*
     *  Calculates the number of elements int the matrix
     *  @return Number of elements in the matrix
    */
    __host__ __device__ inline size_t size() {
        return (size_t)width * height;
    }

    /*
     *  Returns pointer to device object clone (if exists)
     *  @return Pointer to device clone or nullptr if not present
    */
    __host__ __device__ inline map* dev() {
        return d_this;
    }

    #ifdef __CUDA_ARCH__
    /*
     *  Returns pointer to device data
     *  @return Pointer to device data
    */
    __device__ inline T* data() {
        return d_data;
    }
    #else
    /*
     *  Returns pointer to host data
     *  @return Pointer to host data
    */
    __host__ inline T* data() {
        return h_data;
    }
    #endif

    #ifdef __CUDA_ARCH__
    /*
     *  Matrix indexation operator (device)
     *  @param x x-coordinate of the matrix
     *  @param y y-coordinate of the matrix
     *  @return Matrix element on (x y) position
    */
    __device__ inline T& operator() (uint16_t x, uint16_t y) {
        return d_data[(uint32_t)y * width + x];
    }
    #else
    /*
     *  Matrix indexation operator (host)
     *  @param x x-coordinate of the matrix
     *  @param y y-coordinate of the matrix
     *  @return Matrix element on (x y) position
    */
    __host__ inline T& operator() (uint16_t x, uint16_t y) {
        return h_data[(uint32_t)y * width + x];
    }
    #endif

    #ifdef __CUDA_ARCH__
    /*
     *  Matrix indexation operator (device)
     *  @param i index of the C-style array
     *  @return Matrix C-style array element on index [i]
    */
    __device__ inline T& operator[] (uint32_t i) {
        return d_data[i];
    }
    #else
    /*
     *  Matrix indexation operator (host)
     *  @param i index of the C-style array
     *  @return Matrix C-style array element on index [i]
    */
    __host__ inline T& operator[] (uint32_t i) {
        return h_data[i];
    }
    #endif

    /*
     *  Copy assignment operator
     *  @param m source map object
     *  @return left operand after assignment
    */
    __host__ inline map& operator= (const map &m) {
        if (&m == this) return *this;

        ~map();

        width   = m.width;
        height  = m.height;
        type    = m.type;

        switch(type) {
            case NONE:
                h_data = nullptr;
                d_data = nullptr;
                d_this = nullptr;
                break;

            case HOST:
                _alloc_host();
                std::memcpy(h_data, m.h_data, size() * sizeof(T));
                d_data = nullptr;
                d_this = nullptr;
                break;

            case DEVICE:
                _alloc_device();
                cudaMemcpy(d_data, m.d_data, size() * sizeof(T), cudaMemcpyDeviceToDevice);
                h_data = nullptr;
                break;

            case HOST | DEVICE:
                _alloc_host();
                std::memcpy(h_data, m.h_data, size() * sizeof(T));
                _alloc_device();
                cudaMemcpy(d_data, m.d_data, size() * sizeof(T), cudaMemcpyDeviceToDevice);
                break;

            case UNIFIED:
                cudaMallocManaged((void**)&h_data, size() * sizeof(T));
                d_data = h_data;
                cudaMalloc((void**)&d_this, sizeof(*this));
                cudaMemcpy(d_this, this, sizeof(*this), cudaMemcpyHostToDevice);
                std::memcpy(h_data, m.h_data, size() * sizeof(T));
                break;

            default:
                std::cerr << "E: map at " << this << " - invalit memory type\n";
                break;
        }

        return *this;
    }

    /*
     *  Move assignment operator
     *  @param m source map object
     *  @return left operand after assignment
    */
    __host__ inline map& operator= (const map &&m) {
        if (&m == this) return *this;

        ~map();

        width       = m.width;
        height      = m.height;

        h_data      = m.h_data;
        m.h_data    = nullptr;

        d_this      = m.d_this;
        m.d_this    = nullptr;

        d_data      = m.d_data;
        m.d_data    = nullptr;

        m.type      = NONE;

        return *this;
    }

    /*
     *  Move assignment operator
     *  @param type memory allocation type
     *  @return void
    */
    __host__ inline void alloc(uint8_t type) {
        switch(type) {
            case NONE:
                std::cerr << "W: map at " << this << " - could no allocate type NONE\n";
                break;

            case HOST:
                _alloc_host();
                break;

            case DEVICE:
                _alloc_device();
                break;

            case HOST | DEVICE:
                _alloc_host();
                _alloc_device();
                break;

            case UNIFIED:
                cudaMallocManaged((void**)&h_data, size() * sizeof(T));
                d_data = h_data;
                cudaMalloc((void**)&d_this, sizeof(*this));
                cudaMemcpy(d_this, this, sizeof(*this), cudaMemcpyHostToDevice);
                break;

            default:
                std::cerr << "\033[31m" << "E: map at " << this << " - invalit memory type\n" << "\033[0m";
                break;
        }
    }

    /*
     *  Transfers matrix data between CPU and GPU memory
     *  @param type transfer direction type
     *  @return void
    */
    __host__ inline void transfer(TRANSFER_TYPE type) {
        switch(type) {
            case D2H:
                cudaMemcpy(h_data, d_data, size() * sizeof(T), cudaMemcpyDeviceToHost);
                break;
            case H2D:
                cudaMemcpy(d_data, h_data, this->size() * sizeof(T), cudaMemcpyHostToDevice);
                break;
            default:
                std::cerr << "\033[31m" << "E: map at address " << this << " - invalid transfer direction\n" << "\033[0m";
                break;
        }
    }

    /*
     *  Destructor
    */
    __host__ inline ~map() {
        switch(type) {
            case NONE:
                break;

            case HOST:
                delete[] h_data;
                break;

            case DEVICE:
                cudaFree(d_this);
                cudaFree(d_data);
                break;

            case HOST | DEVICE:
                delete[] h_data;
                cudaFree(d_this);
                cudaFree(d_data);
                break;

            case UNIFIED:
                cudaFree(h_data);
                break;

            default:
                std::cerr << "\033[31m" << "E: map at " << this << " - invalit memory type\n" << "\033[0m";
                break;
        }
    }

private:
    __host__ void _alloc_host() {
        h_data = new T[size()];
    }

    __host__ void _alloc_device() {
        cudaMalloc((void**)&d_data, size() * sizeof(T));
        cudaMalloc((void**)&d_this, sizeof(*this));
        cudaMemcpy(d_this, this, sizeof(*this), cudaMemcpyHostToDevice);
    }
};

#else

/*
 *  Class containing a matrix
 *  @param width matrix width
 *  @param height matrix height
 *  @param *data pointer to data
*/
template <typename T> 
class map {
//private:
public:
    uint16_t    width, height;
    T           *data;

public:
    /*
     *  Default constructor
    */
    inline map()
        : width(0), height(0), data(nullptr) {
    }

    /*
     *  Constructor based on the matrix size
     *  @param width width of the matrix
     *  @param height height of the matrix
     *  @param type mep memory type
    */
    inline map(uint16_t width, uint16_t height)
        : width(width), height(height) {
        _alloc();
    }

    /*
     *  Constructor based on other map object (deep copy contructor)
     *  @param m source matrix
    */
    inline map(const map &m)
        : width(m.width), height(m.height) {
        _alloc_host();
        std::memcpy(data, m.h_data, size() * sizeof(T));
    }

    /*
     *  Constructor based on other map object (move contructor)
     *  @param m source map object
    */
    inline map(map &&m) noexcept
        : width(m.width), height(m.height), data(m.data) {
        m.data      = nullptr;
    }

    /*
     *  Calculates the number of elements int the matrix
     *  @return Number of elements in the matrix
    */
    inline size_t size() {
        return (size_t)width * height;
    }

    /*
     *  Returns pointer to host data
     *  @return Pointer to host data
    */
    inline T* data() {
        return data;
    }

    /*
     *  Matrix indexation operator
     *  @param x x-coordinate of the matrix
     *  @param y y-coordinate of the matrix
     *  @return Matrix element on (x y) position
    */
    inline T& operator() (uint16_t x, uint16_t y) {
        return data[(uint32_t)y * width + x];
    }

    /*
     *  Matrix indexation operator
     *  @param i index of the C-style array
     *  @return Matrix C-style array element on index [i]
    */
    inline T& operator[] (uint32_t i) {
        return data[i];
    }

    /*
     *  Copy assignment operator
     *  @param m source map object
     *  @return left operand after assignment
    */
    inline map& operator= (const map &m) {
        if (&m == this) return *this;

        ~map();

        width   = m.width;
        height  = m.height;

        _alloc_host();
        std::memcpy(h_data, m.data, size() * sizeof(T));

        return *this;
    }

    /*
     *  Move assignment operator
     *  @param m source map object
     *  @return left operand after assignment
    */
    inline map& operator= (const map &&m) {
        if (&m == this) return *this;

        ~map();

        width       = m.width;
        height      = m.height;

        data        = m.data;
        m.data      = nullptr;

        return *this;
    }

    /*
     *  Destructor
    */
    inline ~map() {
        _free();
    }

private:
    inline void _alloc() {
        data = new T[size()];
    }

    inline void _free() {
        delete[] data;
    }
};