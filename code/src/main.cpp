#include <iostream>

#include "tensor.hpp"

int main(int argc, char** argv) {
    const auto data = Tensor<int32_t>{Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};

    std::cout << data.shape() << std::endl;
    for (auto&& x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
