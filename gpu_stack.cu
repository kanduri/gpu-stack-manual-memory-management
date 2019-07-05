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

    template <typename T, typename F>
    __global__
    void push_back_F(stack_impl<T>* s, F f) {
        if (f(threadIdx.x)) {
            push(s, int(threadIdx.x));
        }
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
    stack<T> s(capacity);

    // 2. Push back a data point
    auto impl_p = s.get_impl();
    T value = 42;
    kernels::push_back<<<1,1>>>(impl_p, value);

        // 3. Get data back and confirm size/data
    auto data_v = s.get_and_clear();

    EXPECT_EQ(1u, data_v.size());
    EXPECT_EQ(value, data_v[0]);

}

TEST(stack, push_back) {

    using T = int;

    const unsigned n = 10;
    EXPECT_TRUE(n%2 == 0); // require n is even for tests to work

    stack<T> s(n);
    auto impl_p = s.get_impl();

    kernels::push_back_F<<<1, n>>>(impl_p, kernels::all_ftor());
    cudaDeviceSynchronize();
    auto data_v = s.get_and_clear();
    EXPECT_EQ(n, data_v.size());
    std::sort(data_v.begin(), data_v.end());
    for (auto i=0; i<int(data_v.size()); ++i) {
        EXPECT_EQ(i, data_v[i]);
    }

    kernels::push_back_F<<<1, n>>>(impl_p, kernels::even_ftor());
    cudaDeviceSynchronize();
    data_v = s.get_and_clear();
    EXPECT_EQ(n/2, data_v.size());
    std::sort(data_v.begin(), data_v.end());
    for (auto i=0; i<int(data_v.size()); ++i) {
        EXPECT_EQ(2*i, data_v[i]);
    }

    kernels::push_back_F<<<1, n>>>(impl_p, kernels::odd_ftor());
    cudaDeviceSynchronize();
    data_v = s.get_and_clear();
    EXPECT_EQ(n/2, data_v.size());
    std::sort(data_v.begin(), data_v.end());
    for (auto i=0; i<int(data_v.size()); ++i) {
        EXPECT_EQ(2*i+1, data_v[i]);
    }
}

TEST(stack, empty) {

    using T = int;

    stack<T> s(0u);
    
    // get copy of stack storage from device
    auto impl_copy = s.get_impl_copy();

    EXPECT_EQ(0u, impl_copy.size);
    EXPECT_EQ(0u, impl_copy.capacity);

    EXPECT_EQ(impl_copy.data, nullptr);

    // push a value into an empty stack and check
    T value = 42;
    auto impl_p = s.get_impl();
    kernels::push_back<<<1,1>>>(impl_p, value);
    cudaDeviceSynchronize();

    // get copy of stack storage from device
    impl_copy = s.get_impl_copy();
    EXPECT_EQ(1u, impl_copy.size);
}

TEST(stack, overflow) {

    using T = int;

    const unsigned n = 10;

    stack<T> s(n);
    auto impl_p = s.get_impl();

    kernels::push_back_F<<<1, 2*n>>>(impl_p, kernels::all_ftor()); 
    cudaDeviceSynchronize();

    EXPECT_EQ(n, s.size());
    EXPECT_EQ(2*n, s.pushes());

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}