#include <vector>
#include <stdexcept>

#include "gtest/gtest.h"
#include "gpu_stack.hpp"
#include "host_stack.hpp"

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}