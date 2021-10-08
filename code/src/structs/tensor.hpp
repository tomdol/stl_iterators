#pragma once

#include "shape.hpp"
#include "validation.hpp"
#include <memory>

template <typename T>
struct Tensor {
    Tensor(Shape shape) : _shape{std::move(shape)}, _buffer{std::make_unique<T[]>(_shape.shape_capacity())} {}

    Tensor(Shape shape, std::initializer_list<T> values) : Tensor<T>{std::move(shape)} {
        std::copy(std::begin(values), std::end(values), _buffer.get());
    }

    Tensor(Shape shape, const std::vector<T>& values) : Tensor<T>{std::move(shape)} {
        std::copy(std::begin(values), std::end(values), _buffer.get());
    }

    ~Tensor() = default;

    void reset() { _buffer = std::make_unique<T[]>(_shape.shape_capacity()); }

    T* buffer() { return _buffer.get(); }
    const T* const buffer() const { return _buffer.get(); }
    size_t elements() const { return _shape.shape_capacity(); }

    const Shape& shape() const noexcept { return _shape; }

    friend std::ostream& operator<<(std::ostream& s, const Tensor& t) {
        const auto& shape = t.shape();
        THROW_IF(shape.rank() != 4, "This operator can only output 4D tensors");

        size_t elem_idx = 0;
        for (size_t b = 0; b < shape[0]; ++b) {
            for (size_t c = 0; c < shape[1]; ++c) {
                for (size_t row = 0; row < shape[2]; ++row) {
                    s << "  [";
                    for (size_t col = 0; col < shape[3]; ++col) {
                        const T elem = *(t.buffer() + elem_idx++);
                        const auto onechar = std::abs(elem) < 10;
                        const auto negative = elem < 0;
                        size_t padding_len = 1;
                        if (onechar)
                            padding_len += 1;
                        if (negative)
                            padding_len -= 1;
                        const auto padding = std::string(padding_len, ' ');
                        s << padding << elem << ", ";
                    }
                    s << "\b\b]" << std::endl;
                }
            }
        }

        return s;
    }

  private:
    Shape _shape;
    std::unique_ptr<T[]> _buffer;
};
