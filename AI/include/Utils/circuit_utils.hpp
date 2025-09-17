#ifndef CIRCUIT_UTILS_HPP
#define CIRCUIT_UTILS_HPP

#include <array>
#include <limits>

namespace ai_pass_selector {
    constexpr std::array<unsigned int, 3> TINY_CIRCUIT_SPECS = {5, 25, 10};
    constexpr std::array<unsigned int, 3> SMALL_CIRCUIT_SPECS = {20, 500, 50};
    constexpr std::array<unsigned int, 3> MODERATE_CIRCUIT_SPECS = {100, 1000, 200};
    constexpr std::array<unsigned int, 3> BIG_CIRCUIT_SPECS = {250, 30000, 250};
    constexpr std::array<unsigned int, 3> HUGE_CIRCUIT_SPECS = {
        std::numeric_limits<unsigned int>::max(),
        std::numeric_limits<unsigned int>::max(),
        std::numeric_limits<unsigned int>::max()
    };
} // namespace ai_pass_selector
#endif // CIRCUIT_UTILS_HPP
