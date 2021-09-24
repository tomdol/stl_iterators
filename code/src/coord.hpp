#pragma once

#include "shape.hpp"

#include <algorithm>
#include <array>

template <size_t Dims, typename T = int>
struct Coord : std::array<T, Dims> {
    Coord() = default;

    template <typename U>
    Coord(const std::initializer_list<U>& values) : std::array<T, Dims>{} {
        if constexpr (std::is_unsigned_v<U>) {
            // static_cast to avoid narrowing conversion warnings when constructing with size_t elements
            std::transform(std::begin(values), std::end(values), this->begin(), [](U u) { return static_cast<T>(u); });
        } else {
            std::copy(std::begin(values), std::end(values), this->begin());
        }
    }

    Coord(const Coord<Dims>& other, const Shape& paddings_begin) : std::array<T, Dims>{} {
        for (size_t i = 0; i < other.size(); ++i) {
            this->operator[](i) = other[i] - paddings_begin[i];
        }
    }

    friend std::ostream& operator<<(std::ostream& s, const Coord<Dims, T>& c) {
        s << "Coord{";
        for (size_t i = 0; i < c.size(); ++i) {
            s << c[i] << ",";
        }
        s << "\b}";
        return s;
    }
};

using Coord2D = Coord<2>;
