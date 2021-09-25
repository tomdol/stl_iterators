#include <iostream>

#include "max_pool.hpp"
#include "tensor.hpp"

int main(int argc, char** argv) {
    const auto data = Tensor<int32_t>{Shape{1, 1, 4, 4}, {25, 5, 16, 27, -7, 13, -2, 7, 17, 4, 22, 28, -20, 11, 0, 17}};
    auto output = Tensor<int32_t>{Shape{1, 1, 3, 3}};

    std::cout << data << std::endl;

    max_pool(data, output, Shape{2, 2}, Shape{0, 0}, Shape{0, 0});

    std::cout << output << std::endl;

    return 0;
}
