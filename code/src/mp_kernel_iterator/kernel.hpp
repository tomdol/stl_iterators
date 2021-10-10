#pragma once

#include "structs/coord.hpp"
#include "structs/shape.hpp"

#include <span>

struct EndIteratorTag {};
template <typename T>
struct KernelIterator final {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;

    KernelIterator(T* tensor_data, const Coord2D& kernel_position, const Shape& kernel_shape, const Shape& tensor_shape,
                   const Shape& dilations) :
        _tensor_data{tensor_data},
        _kernel_cols{kernel_shape[1]},
        _tensor_columns_count{tensor_shape[3]},
        _row_dilation{dilations[0]},
        _col_dilation{dilations[1]} {

        _base_offset = _tensor_columns_count * kernel_position[0] + kernel_position[1];
        _data_elem_idx = _base_offset;
    }

    KernelIterator(const Coord2D& kernel_position, const Shape& kernel_shape, const Shape& tensor_shape,
                   const Shape& dilations, EndIteratorTag) {
        _data_elem_idx = tensor_shape[3] * (kernel_position[0] + kernel_shape[0] * dilations[0]);
        _data_elem_idx += kernel_position[1];
    }

    T& operator*() { return _tensor_data[_data_elem_idx]; }
    const T& operator*() const { return *(_tensor_data + _data_elem_idx); }

    bool operator==(const KernelIterator<T>& other) const { return _data_elem_idx == other._data_elem_idx; }

    KernelIterator<T>& operator++() {
        if (++_kernel_col == _kernel_cols) {
            _kernel_col = 0;
            ++_kernel_row;
        }

        _data_elem_idx =
            _base_offset + _kernel_row * _row_dilation * _tensor_columns_count + _kernel_col * _col_dilation;

        return *this;
    }

  private:
    T* _tensor_data = nullptr;
    int32_t _kernel_cols;
    int32_t _tensor_columns_count = 0;
    int32_t _col_dilation = 1;
    int32_t _row_dilation = 1;

    int32_t _base_offset = 0;
    int32_t _data_elem_idx = 0;

    int32_t _kernel_row = 0;
    int32_t _kernel_col = 0;
};

template <typename T>
struct Kernel final {
    Kernel(const Shape& kernel_shape, const Coord2D& kernel_position, const Shape& tensor_shape, T* tensor_data) :
        _kernel_shape{kernel_shape},
        _kernel_position{kernel_position},
        _dilations{1, 1},
        _tensor_shape{tensor_shape},
        _tensor_data{tensor_data} {}

    Kernel(const Shape& kernel_shape, const Coord2D& kernel_position, const Shape& dilations, const Shape& tensor_shape,
           T* tensor_data) :
        _kernel_shape{kernel_shape},
        _kernel_position{kernel_position},
        _dilations{dilations},
        _tensor_shape{tensor_shape},
        _tensor_data{tensor_data} {}

    KernelIterator<T> begin() const {
        return KernelIterator<T>(_tensor_data, _kernel_position, _kernel_shape, _tensor_shape, _dilations);
    }

    KernelIterator<T> end() const {
        return KernelIterator<T>(_kernel_position, _kernel_shape, _tensor_shape, _dilations, EndIteratorTag{});
    }

  private:
    Shape _kernel_shape;
    Coord2D _kernel_position;
    Shape _dilations;
    Shape _tensor_shape;
    T* _tensor_data;
};
