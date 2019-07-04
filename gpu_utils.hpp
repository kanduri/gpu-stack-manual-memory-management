#pragma once

#include "cuda.h"
#include "cuda_runtime_api.h"

// gpu_utils.hpp

template <typename T>
T* gpu_malloc(size_t n) {

    T* tmp;
    auto status = cudaMalloc(&tmp, n*sizeof(T));

    if (status != 0) {
        throw std::runtime_error(cudaGetErrorName(status));
    }

    return tmp;
}

template <typename T>
void h2d_mem_copy(T* destination, T* source, size_t size) {

    auto status = cudaMemcpy(destination, source, size, cudaMemcpyHostToDevice);

    if (status != 0) {
        throw std::runtime_error(cudaGetErrorName(status));
    }
}

template <typename T>
void d2h_mem_copy(T* destination, T* source, size_t size) {

    auto status = cudaMemcpy(destination, source, size, cudaMemcpyDeviceToHost);

    if (status != 0) {
        throw std::runtime_error(cudaGetErrorName(status));
    }
}