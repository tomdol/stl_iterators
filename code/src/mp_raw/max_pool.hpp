#pragma once

#include "structs/coord.hpp"
#include "structs/shape.hpp"
#include "structs/tensor.hpp"
#include "structs/validation.hpp"

#include <limits>

namespace mp_raw {
template <typename T>
void max_pool(const Tensor<T>& data, Tensor<T>& out, const Shape& kernel, const Shape& paddings_begin) {
    const Shape& data_shape = data.shape();
    const Shape& out_shape = out.shape();
    THROW_IF(data_shape.rank() != 4 || out_shape.rank() != 4, "This code only supports 4D tensors.");

    size_t out_idx = 0;
    for (size_t b = 0; b < data_shape[0]; ++b) {
        for (size_t c = 0; c < data_shape[1]; ++c) {
            for (size_t out_row = 0u; out_row < out_shape[2]; ++out_row) {
                for (size_t out_col = 0u; out_col < out_shape[3]; ++out_col) {
                    T max_elem = std::numeric_limits<T>::lowest();

                    const auto out_elem_coord = Coord2D({out_row, out_col});
                    const auto kernel_position = Coord2D(out_elem_coord, paddings_begin);

                    for (size_t kernel_row = 0; kernel_row < kernel[0]; ++kernel_row) {
                        for (size_t kernel_col = 0; kernel_col < kernel[1]; ++kernel_col) {
                            const size_t data_elem_index =
                                data_shape[3] * (kernel_row + kernel_position[0]) + kernel_col + kernel_position[1];

                            const auto current_elem = *(data.buffer() + data_elem_index);
                            if (current_elem > max_elem) {
                                max_elem = current_elem;
                            }
                        }
                    }

                    *(out.buffer() + out_idx) = max_elem;
                    ++out_idx;
                }
            }
        }
    }
}
} // namespace mp_raw
