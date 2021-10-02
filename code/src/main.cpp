#include <iostream>

#include "mp_kernel_iterator/max_pool.hpp"
#include "mp_raw/max_pool.hpp"
#include "structs/tensor.hpp"

void infer_max_pool_raw(const Tensor<int32_t>& data, Tensor<int32_t>& output, const Shape& kernel,
                        const Shape& pads_begin, const Shape& pads_end) {
    mp_raw::max_pool(data, output, kernel, pads_begin);
}

void infer_max_pool_iter(const Tensor<int32_t>& data, Tensor<int32_t>& output, const Shape& kernel,
                         const Shape& pads_begin, const Shape& pads_end) {
    mp_iter::max_pool(data.buffer(), output.buffer(), data.shape(), output.shape(), kernel, pads_begin, pads_end);
}

int main(int argc, char** argv) {
    // clang-format off
    const auto data = Tensor<int32_t>{Shape{1, 1, 4, 4}, 
        {25,  5, 16, 27, 
         -7, 13, -2,  7, 
         17,  4,  0, 16, 
         29, 11, 22, 28}};
    // clang-format on

    auto output = Tensor<int32_t>{Shape{1, 1, 3, 3}};
    const auto kernel = Shape{2, 2};
    const auto pads_begin = Shape{0, 0};
    const auto pads_end = Shape{0, 0};

    std::cout << data << std::endl;

    infer_max_pool_raw(data, output, kernel, pads_begin, pads_end);
    std::cout << output << std::endl;

    output.reset();
    infer_max_pool_iter(data, output, kernel, pads_begin, pads_end);
    std::cout << output << std::endl;

    return 0;
}
