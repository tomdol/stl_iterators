#pragma once

#include "kernel.hpp"
#include "structs/coord.hpp"
#include "structs/shape.hpp"
#include "structs/tensor.hpp"
#include "structs/validation.hpp"

template <typename T>
void max_pool(const T* data, T* output, const Shape& data_shape, const Shape& out_shape, const Shape& kernel_shape,
              const Shape& paddings_begin) {
    THROW_IF(data_shape.rank() != 4 || out_shape.rank() != 4, "This code only supports 4D tensors.");

    const auto batch_elems = data_shape.sub_shape(1).shape_capacity();
    const auto channel_elems = data_shape.sub_shape(2).shape_capacity();

    for (size_t b = 0; b < data_shape[0]; ++b) {
        for (size_t c = 0; c < data_shape[1]; ++c) {
            const T* channel_data = data + b * batch_elems + c * channel_elems;
            T* out_channel_data = output; // + calculated output channel offset

            auto out_shape_begin = out_shape.begin();
            auto out_shape_end = out_shape.end();

            std::transform(out_shape_begin, out_shape_end, out_channel_data, [&](const Coord2D& out_elem) {
                const auto kernel_position = Coord2D({out_elem[0], out_elem[1]}, paddings_begin);
                const auto kernel = Kernel{kernel_shape, kernel_position, data_shape, channel_data};

                return std::max_element(std::begin(kernel), std::end(kernel));
            });
        }
    }
}
