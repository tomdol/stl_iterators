#pragma once

#include <numeric>
#include <vector>

struct Shape : std::vector<uint32_t> {
    using dimension_t = value_type;

    Shape(std::initializer_list<dimension_t> dims) : std::vector<dimension_t>{std::begin(dims), std::end(dims)} {}

    uint32_t shape_capacity() const { return std::accumulate(begin(), end(), 1, std::multiplies<uint32_t>()); }
};

std::ostream& operator<<(std::ostream& s, const Shape& shape) {
    s << "Shape{";
    for (size_t i = 0; i < shape.size(); ++i) {
        s << shape[i] << ",";
    }
    s << "\b}";
    return s;
}
