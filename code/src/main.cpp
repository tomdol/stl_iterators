#include <iostream>

#include "mp_kernel_iterator/max_pool.hpp"
#include "mp_raw/max_pool.hpp"
#include "structs/tensor.hpp"

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

    mp_raw::max_pool(data, output, kernel, pads_begin);
    std::cout << output << std::endl;

    output.reset();
    mp_iter::max_pool(data.buffer(), output.buffer(), data.shape(), output.shape(), kernel, pads_begin);
    std::cout << output << std::endl;

    auto strides = Shape{2, 2};
    auto dilations = Shape{1, 1};

    auto output2x2 = Tensor<int32_t>{Shape{1, 1, 2, 2}};
    mp_raw::ext_max_pool(data, output2x2, kernel, pads_begin, strides, dilations);
    std::cout << output2x2 << std::endl;

    output2x2.reset();
    mp_iter::ext_max_pool(data.buffer(), output2x2.buffer(), data.shape(), output2x2.shape(), kernel, pads_begin,
                          strides, dilations);
    std::cout << output2x2 << std::endl;

    strides = Shape{1, 1};
    dilations = Shape{2, 2};

    output2x2.reset();
    mp_raw::ext_max_pool(data, output2x2, kernel, pads_begin, strides, dilations);
    std::cout << output2x2 << std::endl;

    output2x2.reset();
    mp_iter::ext_max_pool(data.buffer(), output2x2.buffer(), data.shape(), output2x2.shape(), kernel, pads_begin,
                          strides, dilations);
    std::cout << output2x2 << std::endl;

    return 0;
}
