#pragma once

#include "coord.hpp"
#include "shape.hpp"

#include <span>

template <typename T>
struct KernelIterator {
    KernelIterator(std::span<T> tensor_data, const Shape& kernel_shape, const Shape& tensor_shape) :
        _tensor_data{std::move(tensor_data)},
        _span_elem{std::begin(_tensor_data)},
        _kernel_shape{kernel_shape},
        _tensor_shape{tensor_shape} {}

    T& operator*() { return *(_tensor_data.begin()); }
    const T& operator*() const { return *(_tensor_data.begin()); }

    bool operator==(const KernelIterator<T>& other) const { return false; }

    KernelIterator<T>& operator++() { return *this; }

  private:
    std::span<T> _tensor_data;
    std::span<T>::iterator _span_elem;
    Shape _kernel_shape;
    Shape _tensor_shape;
};

template <typename T>
struct Kernel {
    Kernel(const Shape& kernel_shape, const Coord2D& kernel_position, const Shape& tensor_shape,
           std::span<T> tensor_data) :
        _kernel_shape{kernel_shape},
        _kernel_position{kernel_position},
        _tensor_shape{tensor_shape},
        _tensor_data{tensor_data} {}

    KernelIterator<T> begin() const { return KernelIterator<T>{_tensor_data, _kernel_shape, _tensor_shape}; }
    KernelIterator<T> end() const { return KernelIterator<T>{_tensor_data, _kernel_shape, _tensor_shape}; }

  private:
    const Shape _kernel_shape;
    const Coord2D _kernel_position;
    const Shape _tensor_shape;
    const std::span<T> _tensor_data;
};
