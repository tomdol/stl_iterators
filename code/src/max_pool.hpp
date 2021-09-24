#pragma once

#include "coord.hpp"
#include "shape.hpp"
#include "tensor.hpp"
#include "validation.hpp"

#include <limits>

template <typename T>
void max_pool(const Tensor<T>& data, Tensor<T>& out, const Shape& kernel, const Shape& paddings_begin,
              const Shape& paddings_end) {
    const Shape& data_shape = data.shape();
    const Shape& out_shape = out.shape();
    THROW_IF(data_shape.size() != 4 || out_shape.size() != 4, "This code only supports 4D tensors.");

    const auto elements_in_channel = data_shape[2] * data_shape[3];
    const auto elements_in_batch = elements_in_channel * data_shape[1];

    const auto elements_in_out_channel = out_shape[2] * out_shape[3];
    const auto elements_in_out_batch = elements_in_out_channel * out_shape[1];

    size_t out_idx = 0;
    for (size_t b = 0; b < data_shape[0]; ++b) {
        for (size_t c = 0; c < data_shape[1]; ++c) {
            const T* first_elem_in_channel = data.begin() + b * elements_in_batch + c * elements_in_channel;
            T* first_elem_in_out_channel = out.begin() + b * elements_in_out_batch + c * elements_in_out_channel;

            for (size_t out_row = 0u; out_row < out_shape[2]; ++out_row) {
                for (size_t out_col = 0u; out_col < out_shape[3]; ++out_col) {
                    T max_elem = std::numeric_limits<T>::lowest();

                    const auto out_elem_coord = Coord2D({out_row, out_col});
                    const auto kernel_position = Coord2D(out_elem_coord, paddings_begin);

                    for (size_t kernel_row = 0; kernel_row < kernel[0]; ++kernel_row) {
                        for (size_t kernel_col = 0; kernel_col < kernel[1]; ++kernel_col) {
                            const auto kernel_elem_coord = Coord2D({kernel_row, kernel_col});
                            const size_t data_elem_index = data_shape[3] * (kernel_elem_coord[0] + kernel_position[0]) +
                                                           kernel_elem_coord[1] + kernel_position[1];

                            if (*(data.begin() + data_elem_index) > max_elem) {
                                max_elem = *(data.begin() + data_elem_index);
                            }
                        }
                    }

                    *(out.begin() + out_idx) = max_elem;
                    ++out_idx;
                }
            }
        }
    }
}
