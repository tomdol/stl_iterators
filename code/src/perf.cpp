#include <algorithm>
#include <chrono>
#include <iomanip>
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
    mp_iter::max_pool(data.buffer(), output.buffer(), data.shape(), output.shape(), kernel, pads_begin);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main(int argc, char** argv) {
    const auto spatial = 1500;
    const auto in_shape = Shape{1, 1, spatial, spatial};
    const auto out_shape = Shape{1, 1, spatial - 1, spatial - 1};

    std::vector<int32_t> input_values;
    input_values.resize(in_shape.shape_capacity());
    std::iota(input_values.begin(), input_values.end(), 1);
    // std::random_shuffle(std::begin(input_values), std::end(input_values));
    const auto data = Tensor<int32_t>(in_shape, input_values);
    auto output = Tensor<int32_t>{out_shape};

    const auto kernel = Shape{2, 2};
    const auto pads_begin = Shape{0, 0};
    const auto pads_end = Shape{0, 0};

    const auto elapsed_time_iter = infer_max_pool_iter(data, output, kernel, pads_begin, pads_end);
    std::cout << "Elapsed time (iter): " << elapsed_time_iter << "ms" << std::endl;
    output.reset();

    const auto elapsed_time_raw = infer_max_pool_raw(data, output, kernel, pads_begin, pads_end);
    // std::cout << output << std::endl;
    std::cout << "Elapsed time  (raw): " << elapsed_time_raw << "ms" << std::endl;
    output.reset();

    std::cout << "\niter / raw: " << std::setprecision(2) << float(elapsed_time_iter) / float(elapsed_time_raw)
              << "x slower" << std::endl;

    return 0;
}
