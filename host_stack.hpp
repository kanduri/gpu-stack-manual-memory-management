#pragma once

#include "stack_storage.hpp"
#include "gpu_utils.hpp"

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

    stack_impl<T>* get_impl() {
        return impl_;
    }

    size_t size() {
        return impl_->size;
    }
};