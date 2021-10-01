#pragma once

#include "coord.hpp"
#include "shape.hpp"

#include <span>

struct EndIteratorTag {};
template <typename T>
struct KernelIterator final {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;

    KernelIterator(const std::span<T>& tensor_data, const Coord2D& kernel_position, const Shape& kernel_shape,
                   const Shape& tensor_shape) :
        _tensor_data{tensor_data}, _kernel_shape{kernel_shape}, _tensor_columns_count{tensor_shape[3]} {

        _base_offset = _tensor_columns_count * kernel_position[0] + kernel_position[1];
        _data_elem_idx = _base_offset;
    }

    KernelIterator(const Coord2D& kernel_position, const Shape& kernel_shape, const Shape& tensor_shape, EndIteratorTag) {
        _data_elem_idx = tensor_shape[3] * kernel_position[0] + kernel_position[1];
        _data_elem_idx += tensor_shape[3] * kernel_shape[1];
    }

    reference operator*() { return _tensor_data[_data_elem_idx]; }
    const reference operator*() const { return _tensor_data[_data_elem_idx]; }

    bool operator==(const KernelIterator<T>& other) const 
    { 
        return _data_elem_idx == other._data_elem_idx; 
    }

    KernelIterator<T>& operator++() {
        if (++_kernel_col == _kernel_shape[1]) {
            _kernel_col = 0;
            ++_kernel_row;
        }

        _data_elem_idx = _base_offset + _kernel_row * _tensor_columns_count + _kernel_col;

        return *this;
    }

  private:
    std::span<T> _tensor_data;
    Shape _kernel_shape;
    int32_t _tensor_columns_count = 0;

    int32_t _base_offset = 0;
    int32_t _data_elem_idx = 0;

    int32_t _kernel_row = 0;
    int32_t _kernel_col = 0;
};

template <typename T>
struct Kernel final {
    Kernel(const Shape& kernel_shape, const Coord2D& kernel_position, const Shape& tensor_shape,
           const std::span<T>& tensor_data) :
        _kernel_shape{kernel_shape},
        _kernel_position{kernel_position},
        _tensor_shape{tensor_shape},
        _tensor_data{tensor_data} {}

    KernelIterator<T> begin() const {
        return KernelIterator<T>(_tensor_data, _kernel_position, _kernel_shape, _tensor_shape);
    }

    KernelIterator<T> end() const {
        return KernelIterator<T>(_kernel_position, _kernel_shape, _tensor_shape, EndIteratorTag{});
    }

  private:
    Shape _kernel_shape;
    Coord2D _kernel_position;
    Shape _tensor_shape;
    std::span<T> _tensor_data;
};
