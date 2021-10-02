#include <chrono>
#include <iostream>

#include "mp_kernel_iterator/max_pool.hpp"
#include "mp_raw/max_pool.hpp"
#include "structs/tensor.hpp"

using elapsed_time_t = std::chrono::high_resolution_clock::duration::rep;

elapsed_time_t infer_max_pool_raw(const Tensor<int32_t>& data, Tensor<int32_t>& output, const Shape& kernel,
                                  const Shape& pads_begin, const Shape& pads_end) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    mp_raw::max_pool(data, output, kernel, pads_begin);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

elapsed_time_t infer_max_pool_iter(const Tensor<int32_t>& data, Tensor<int32_t>& output, const Shape& kernel,
                                   const Shape& pads_begin, const Shape& pads_end) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    mp_iter::max_pool(data.buffer(), output.buffer(), data.shape(), output.shape(), kernel, pads_begin, pads_end);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main(int argc, char** argv) {
    std::vector<int32_t> input_values;
    input_values.resize(1000000);
    std::iota(input_values.begin(), input_values.end(), 1);
    const auto data = Tensor<int32_t>(Shape{1, 1, 1000, 1000}, input_values);
    auto output = Tensor<int32_t>{Shape{1, 1, 999, 999}};

    const auto kernel = Shape{2, 2};
    const auto pads_begin = Shape{0, 0};
    const auto pads_end = Shape{0, 0};

    // std::cout << data << std::endl;
    elapsed_time_t elapsed_time = 0;

    elapsed_time = infer_max_pool_iter(data, output, kernel, pads_begin, pads_end);
    // std::cout << output << std::endl;
    std::cout << "Elapsed time: " << elapsed_time << "ms" << std::endl;
    output.reset();

    elapsed_time = infer_max_pool_raw(data, output, kernel, pads_begin, pads_end);
    // std::cout << output << std::endl;
    std::cout << "Elapsed time: " << elapsed_time << "ms" << std::endl;
    output.reset();

    return 0;
}