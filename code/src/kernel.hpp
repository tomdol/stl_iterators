#pragma once

#include "coord.hpp"
#include "shape.hpp"

#include <span>

template <typename T>
struct Kernel {
    Kernel(const Shape& kernel_shape, const Coord2D& kernel_position, const Shape& tensor_shape,
           std::span<T> tensor_data)
        : _kernel_shape{kernel_shape}, _kernel_position{kernel_position}, _tensor_shape{tensor_shape},
          _tensor_data{tensor_data} {}

  private:
    const Shape _kernel_shape;
    const Coord2D _kernel_position;
    const Shape _tensor_shape;
    const std::span<T> _tensor_data;
};
