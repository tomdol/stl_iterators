#include <iostream>

#include "max_pool.hpp"
#include "tensor.hpp"

void infer_max_pool(const Tensor<int32_t>& data, Tensor<int32_t>& output, const Shape& kernel, const Shape& pads_begin,
                    const Shape& pads_end) {
    max_pool(data, output, kernel, pads_begin, pads_end);
}

int main(int argc, char** argv) {
    // clang-format off
    const auto data = Tensor<int32_t>{Shape{1, 1, 4, 4}, 
        {25,  5, 16, 27, 
         -7, 13, -2,  7, 
         17,  4, 22, 28, 
        -20, 11,  0, 17}};
    // clang-format on

    auto output = Tensor<int32_t>{Shape{1, 1, 3, 3}};
    const auto kernel = Shape{2, 2};
    const auto pads_begin = Shape{0, 0};
    const auto pads_end = Shape{0, 0};

    std::cout << data << std::endl;

    infer_max_pool(data, output, kernel, pads_begin, pads_end);
    std::cout << output << std::endl;

    return 0;
}
