#pragma once

#include <iostream>

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