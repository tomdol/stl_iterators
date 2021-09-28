#pragma once

#include <numeric>
#include <vector>

struct Shape {
    using dimension_t = int32_t;

    Shape(std::initializer_list<dimension_t> dims) : _dimensions{std::begin(dims), std::end(dims)} {}

    template <typename Iterator>
    Shape(Iterator first, Iterator last) : _dimensions{first, last} {}

    size_t rank() const { return _dimensions.size(); }

    std::vector<dimension_t>::iterator begin() { return _dimensions.begin(); }
    std::vector<dimension_t>::iterator end() { return _dimensions.end(); }

    std::vector<dimension_t>::const_iterator begin() const { return _dimensions.begin(); }
    std::vector<dimension_t>::const_iterator end() const { return _dimensions.end(); }

    dimension_t shape_capacity() const {
        return std::accumulate(std::begin(_dimensions), std::end(_dimensions), 1, std::multiplies<dimension_t>());
    }

    // TODO: axis validation (out_of_range)
    Shape sub_shape(size_t axis) const { return Shape{std::begin(_dimensions) + axis, std::end(_dimensions)}; }

    dimension_t operator[](size_t i) const { return _dimensions[i]; }

  private:
    std::vector<dimension_t> _dimensions;
};

std::ostream& operator<<(std::ostream& s, const Shape& shape) {
    s << "Shape{";
    for (const auto dim : shape) {
        s << dim << ",";
    }
    s << "\b}";
    return s;
}
