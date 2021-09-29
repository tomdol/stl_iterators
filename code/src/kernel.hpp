#pragma once

#include "coord.hpp"
#include "shape.hpp"

#include <span>

template <typename T>
struct KernelIterator final {
    KernelIterator(const std::span<T>& tensor_data, const Coord2D& kernel_position, const Shape& kernel_shape,
                   const Shape& tensor_shape) :
        _tensor_data{&tensor_data},
        _kernel_position{&kernel_position},
        _kernel_shape{&kernel_shape},
        _tensor_shape{&tensor_shape} {
        _current_kernel_elem = (*_tensor_shape)[3] * (_in_kernel_position[0] + (*_kernel_position)[0]) +
                               _in_kernel_position[1] + (*_kernel_position)[1];
    }

    KernelIterator(const Shape& kernel_shape) { _in_kernel_position[0] = kernel_shape[0] + 1; }

    T& operator*() { return (*_tensor_data)[_current_kernel_elem]; }
    const T& operator*() const { return (*_tensor_data)[_current_kernel_elem]; }

    bool operator==(const KernelIterator<T>& other) const { return _in_kernel_position == other._in_kernel_position; }

    KernelIterator<T>& operator++() {
        if (_in_kernel_position[1] < (*_kernel_shape)[1] - 1) {
            _in_kernel_position[1] += 1;
        } else {
            _in_kernel_position[1] = 0;
            _in_kernel_position[0] += 1;
        }

        _current_kernel_elem = (*_tensor_shape)[3] * (_in_kernel_position[0] + (*_kernel_position)[0]) +
                               _in_kernel_position[1] + (*_kernel_position)[1];
        return *this;
    }

  private:
    const std::span<T>* _tensor_data = nullptr;
    const Coord2D* _kernel_position = nullptr;
    const Shape* _kernel_shape = nullptr;
    const Shape* _tensor_shape = nullptr;

    Coord2D _in_kernel_position{0, 0};
    int32_t _current_kernel_elem = 0;
};

template <typename T>
struct Kernel final {
    Kernel(const Shape& kernel_shape, const Coord2D& kernel_position, const Shape& tensor_shape,
           std::span<T> tensor_data) :
        _kernel_shape{kernel_shape},
        _kernel_position{kernel_position},
        _tensor_shape{tensor_shape},
        _tensor_data{tensor_data} {}

    KernelIterator<T> begin() const {
        return KernelIterator<T>{_tensor_data, _kernel_position, _kernel_shape, _tensor_shape};
    }
    KernelIterator<T> end() const { return KernelIterator<T>{_kernel_shape}; }

  private:
    const Shape _kernel_shape;
    const Coord2D _kernel_position;
    const Shape _tensor_shape;
    const std::span<T> _tensor_data;
};
