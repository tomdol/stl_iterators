#include <iostream>

#include "max_pool.hpp"
#include "tensor.hpp"

int main(int argc, char** argv) {
    const auto data = Tensor<int32_t>{Shape{1, 1, 4, 4}, {25, 5, 16, 27, -7, 27, -2, 7, 17, 4, 22, 28, -20, 11, 0, 17}};
    auto output = Tensor<int32_t>{Shape{1, 1, 3, 3}};

    std::cout << data.shape() << std::endl;
    for (auto&& x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    max_pool(data, output, Shape{0, 0}, Shape{0, 0});

    std::cout << output.shape() << std::endl;
    for (auto&& x : output) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
