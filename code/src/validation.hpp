#pragma once

#include <stdexcept>
#include <string>

void THROW_IF(bool condition, const std::string& message) {
    if (condition) {
        throw std::runtime_error{message};
    }
}
