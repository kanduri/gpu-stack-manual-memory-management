#pragma once

#include "stack_storage.hpp"

#include "cuda.h"
#include "cuda_runtime_api.h"

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