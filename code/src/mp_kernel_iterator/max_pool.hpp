#pragma once

#include "kernel.hpp"
#include "structs/coord.hpp"
#include "structs/shape.hpp"
#include "structs/tensor.hpp"
#include "structs/validation.hpp"

#include <limits>
#include <span>

namespace own {

// doesnt use std::less to compare iterator values
template <class ForwardIterator>
inline ForwardIterator max_element(ForwardIterator __first, ForwardIterator __last) {
    ForwardIterator __i = __first;
    while (++__i != __last)
        if (*__first < *__i)
            __first = __i;

    return __first;
}

} // namespace own

namespace mp_iter {
template <typename T>
void max_pool(const T* data, T* output, const Shape& data_shape, const Shape& out_shape, const Shape& kernel_shape,
              const Shape& paddings_begin, const Shape& paddings_end) {
    THROW_IF(data_shape.rank() != 4 || out_shape.rank() != 4, "This code only supports 4D tensors.");

    const auto batch_elems = data_shape.sub_shape(1).shape_capacity();
    const auto channel_elems = data_shape.sub_shape(2).shape_capacity();

    size_t out_idx = 0;
    for (size_t b = 0; b < data_shape[0]; ++b) {
        for (size_t c = 0; c < data_shape[1]; ++c) {
            const auto channel_offset = b * batch_elems + c * channel_elems;
            const T* channel_data = data + channel_offset;

            for (size_t out_row = 0u; out_row < out_shape[2]; ++out_row) {
                for (size_t out_col = 0u; out_col < out_shape[3]; ++out_col) {
                    const auto kernel_position = Coord2D({out_row, out_col}, paddings_begin);
                    const auto kernel = Kernel{kernel_shape, kernel_position, data_shape, channel_data};

                    const auto max_elem = std::max_element(std::begin(kernel), std::end(kernel));
                    // const auto max_elem = own::max_element(std::begin(kernel), std::end(kernel));

                    *(output + out_idx) = *max_elem;
                    ++out_idx;
                }
            }
        }
    }
}
} // namespace mp_iter

// template <typename T>
// void max_pool(const T* data, T* output, const Shape& data_shape, const Shape& out_shape, const Shape& kernel_shape,
//               const Shape& paddings_begin, const Shape& paddings_end) {
//     THROW_IF(data_shape.rank() != 4 || out_shape.rank() != 4, "This code only supports 4D tensors.");

//     const auto batch_elems = data_shape.sub_shape(1).shape_capacity();
//     const auto channel_elems = data_shape.sub_shape(2).shape_capacity();

//     size_t out_idx = 0;
//     for (size_t b = 0; b < data_shape[0]; ++b) {
//         for (size_t c = 0; c < data_shape[1]; ++c) {
//             const T* channel_data = data + b * batch_elems + c * channel_elems;
//             T* out_channel_data = output; // + calculated output channel offset

//             auto out_shape_begin = out_shape.begin();
//             auto out_shape_begin = out_shape.end();

//             std::transform(out_shape_begin, out_shape_end, out_channel_data, [&](const Coord2D& out_elem) {
//                 const auto kernel_position = Coord2D({out_elem[0], out_elem[1]}, paddings_begin);
//                 const auto kernel = Kernel{kernel_shape, kernel_position, data_shape, channel_data};

//                 return std::max_element(std::begin(kernel), std::end(kernel));
//             });
//         }
//     }
// }
