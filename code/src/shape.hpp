#pragma once

#include <numeric>
#include <vector>

struct Shape : std::vector<int32_t> {
    using dimension_t = value_type;
    // creates an instance representing a scalar (an empty shape)
    Shape() = default;

    Shape(std::initializer_list<dimension_t> dims)
        : std::vector<dimension_t>{std::begin(dims), std::end(dims)} {}

    Shape(const Shape& other) = default;
    Shape(Shape&& other) = default;

    Shape& operator=(const Shape& other) = default;
    Shape& operator=(Shape&& other) = default;

    int32_t shape_capacity() const {
        return std::accumulate(begin(), end(), 1, std::multiplies<int32_t>());
    }
};

std::ostream& operator<<(std::ostream& s, const Shape& shape) {
    s << "Shape{";
    for (size_t i = 0; i < shape.size(); ++i) {
        s << shape[i] << ",";
    }
    s << "\b}";
    return s;
}
