#pragma once

#include "shape.hpp"
#include <memory>

template <typename T>
struct Tensor {
    Tensor(Shape&& shape, std::initializer_list<T> values)
        : _shape{std::move(shape)}, _buffer{std::make_unique<T[]>(_shape.shape_capacity())} {
        std::copy(std::begin(values), std::end(values), _buffer.get());
    }

    const T* begin() const { return _buffer.get(); }
    const T* end() const { return _buffer.get() + _shape.shape_capacity(); }

    const Shape& shape() const noexcept { return _shape; }

  private:
    const Shape _shape;
    std::unique_ptr<T[]> _buffer;
};
