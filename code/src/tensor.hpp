#pragma once

#include "shape.hpp"
#include <memory>

template <typename T>
struct Tensor {
    Tensor(Shape&& shape) : _shape{std::move(shape)}, _buffer{std::make_unique<T[]>(_shape.shape_capacity())} {}

    Tensor(Shape&& shape, std::initializer_list<T> values) : Tensor<T>{std::move(shape)} {
        std::copy(std::begin(values), std::end(values), _buffer.get());
    }

    ~Tensor() = default;

    T* begin() { return _buffer.get(); }
    T* end() { return _buffer.get() + _shape.shape_capacity(); }

    const T* const begin() const { return _buffer.get(); }
    const T* const end() const { return _buffer.get() + _shape.shape_capacity(); }

    const Shape& shape() const noexcept { return _shape; }

  private:
    const Shape _shape;
    std::unique_ptr<T[]> _buffer;
};
