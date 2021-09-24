#pragma once

#include "coord.hpp"
#include "shape.hpp"
#include "tensor.hpp"
#include "validation.hpp"

template <typename T>
void max_pool(const Tensor<T>& data, Tensor<T>& out, const Shape& paddings_begin, const Shape& paddings_end) {
    const Shape& data_shape = data.shape();
    const Shape& out_shape = out.shape();
    THROW_IF(data_shape.size() != 4 || out_shape.size() != 4, "This code only supports 4D tensors.");

    const auto elements_in_channel = data_shape[2] * data_shape[3];
    const auto elements_in_batch = elements_in_channel * data_shape[1];

    const auto elements_in_out_channel = out_shape[2] * out_shape[3];
    const auto elements_in_out_batch = elements_in_out_channel * out_shape[1];

    for (size_t b = 0; b < data_shape[0]; ++b) {
        for (size_t c = 0; c < data_shape[1]; ++c) {
            const T* first_elem_in_channel = data.begin() + b * elements_in_batch + c * elements_in_channel;
            T* first_elem_in_out_channel = out.begin() + b * elements_in_out_batch + c * elements_in_out_channel;

            for (size_t out_row = 0u; out_row < out_shape[2]; ++out_row) {
                for (size_t out_col = 0u; out_col < out_shape[3]; ++out_col) {
                    const auto out_elem_coord = Coord2D({out_row, out_col});
                    const auto kernel_pos = Coord2D(out_elem_coord, paddings_begin);
                    std::cout << kernel_pos << std::endl;
                }
            }
        }
    }
}
