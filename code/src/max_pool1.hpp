#pragma once

#include "coord.hpp"
#include "kernel.hpp"
#include "shape.hpp"
#include "tensor.hpp"
#include "validation.hpp"

#include <limits>
#include <span>

template <typename T>
void max_pool1(const std::span<T> data, std::span<T> output, const Shape& data_shape, const Shape& out_shape,
               const Shape& kernel_shape, const Shape& paddings_begin, const Shape& paddings_end) {
    THROW_IF(data_shape.rank() != 4 || out_shape.rank() != 4, "This code only supports 4D tensors.");

    const auto batch_elems = data_shape.sub_shape(1).shape_capacity();
    const auto channel_elems = data_shape.sub_shape(2).shape_capacity();

    size_t out_idx = 0;
    for (size_t b = 0; b < data_shape[0]; ++b) {
        for (size_t c = 0; c < data_shape[1]; ++c) {
            const auto channel_offset = b * batch_elems + c * channel_elems;
            const auto channel_data = data.subspan(channel_offset, channel_elems);

            for (size_t out_row = 0u; out_row < out_shape[2]; ++out_row) {
                for (size_t out_col = 0u; out_col < out_shape[3]; ++out_col) {
                    const auto kernel_position = Coord2D({out_row, out_col}, paddings_begin);
                    const auto kernel = Kernel{kernel_shape, kernel_position, data_shape, channel_data};

                    const auto max_elem = std::max_element(std::begin(kernel), std::end(kernel));

                    *(output.begin() + out_idx) = *max_elem;
                    ++out_idx;
                }
            }
        }
    }
}
