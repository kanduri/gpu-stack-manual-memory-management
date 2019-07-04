#include <iostream>
#include <vector>
#include <stdexcept>

#include "gtest.h"

#include "cuda.h"
#include "cuda_runtime_api.h"

// stack_storage.hpp

template <typename T>
struct stack_impl {
    unsigned capacity;
    unsigned size;
    T* data;
};

template <typename T>
std::ostream& operator<<(std::ostream& o, stack_impl<T>& s) {
    return o << "<stack_impl: " << s.capacity << ", " << s.size << ", " << s.data << ">";
}

// gpu_stack.hpp
template <typename T>
__device__
void push(stack_impl<T>* s, T v) {
    /// Atomically increment the stores counter. The atomicAdd returns
    // the value of stores before the increment, which is the location
    // at which this thread can store value.
    unsigned position = atomicAdd(&(s->size), 1u);

    // It is possible that stores>capacity. In this case, only capacity
    // entries are stored, and additional values are lost. The stores
    // contains the total number of attempts to push.
    if (position<s->capacity) {
        s->data[position] = v;
    }

    // Note: there are no guards against s.stores overflowing: in which
    // case the stores counter would start again from 0, and values would
    // be overwritten from the front of the stack.
}   

namespace kernels {
    template <typename T>
    __global__
    void push_back(stack_impl<T>* s, T value) {
        push(s, value);
    }

    struct all_ftor {
        __host__ __device__
        bool operator() (int i) {
            return true;
        }
    };

    struct even_ftor {
        __host__ __device__
        bool operator() (int i) {
            return (i%2)==0;
        }
    };

    struct odd_ftor {
        __host__ __device__
        bool operator() (int i) {
            return i%2;
        }
    };
}

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

// host_stack.hpp
template <typename T>
class stack {
    stack_impl<T>* impl_;
    unsigned capacity_;
    
public:

    stack(unsigned c) {
        capacity_ = c;
        // create a stack_impl on the host and complete
        stack_impl<T> impl;
        impl.capacity = capacity_;
        impl.size = 0;
        impl.data = gpu_malloc<T>(capacity_);
        
        // copy the stack_impl to device memory impl_
        impl_ = gpu_malloc<stack_impl<T> >(1);
        h2d_mem_copy<stack_impl<T> >(impl_, &impl, sizeof(impl));
    }   
    
    std::vector<T> get_and_clear() {
        // get a copy of the implementation
        stack_impl<T> impl;
        d2h_mem_copy<stack_impl<T> >(&impl, impl_, sizeof(stack_impl<T>));
        if (impl.size==0u) {
            return {};
        }
        
        // copy the data to host
        std::vector<T> buf(impl.size);
        auto bytes = (impl.size)*sizeof(T);
        d2h_mem_copy(buf.data(), impl.data, bytes);
        
        // reset the implementation size to zero
        impl.size = 0;
        h2d_mem_copy(impl_, &impl, sizeof(impl));
        
        return buf;
    }

    stack_impl<T>* get_impl()
    {
        return impl_;
    }
};

TEST(stack, construction) {

    using T = int;

    unsigned capacity = 10;

    // 1. Create stack on host (and on device) of capacity 10
    stack<T> myStack(capacity);

    // 2. Push back a data point
    auto impl_p = myStack.get_impl();
    T value = 42;
    kernels::push_back<<<1,1>>>(impl_p, value);

        // 3. Get data back and confirm size/data
    auto myData = myStack.get_and_clear();

    EXPECT_EQ(1u, myData.size());
    EXPECT_EQ(value, myData[0]);

}

int main ()
{
    ::testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}