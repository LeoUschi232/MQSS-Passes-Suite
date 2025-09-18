#ifndef CIRCUIT_UTILS_HPP
#define CIRCUIT_UTILS_HPP

#include <string>
#include <limits>
#include <unordered_map>

namespace ai_pass_selector {
    /// Circuit classes
    constexpr int TINY = 1;
    constexpr int SMALL = 2;
    constexpr int MODERATE = 3;
    constexpr int BIG = 4;
    constexpr int HUGE = 5;

    const std::unordered_map<std::string, int>
    CIRCUIT_SIZE_TO_CLASS = {
        {"tiny", TINY},
        {"small", SMALL},
        {"moderate", MODERATE},
        {"big", BIG},
        {"huge", HUGE}
    };

    /// Circuit specifications [max_qubits, max_instructions, max_depth]
    using CircuitSpecs = std::tuple<unsigned int, unsigned int, unsigned int>;
    constexpr CircuitSpecs TINY_CIRCUIT_SPECS = {5, 25, 10};
    constexpr CircuitSpecs SMALL_CIRCUIT_SPECS = {20, 500, 50};
    constexpr CircuitSpecs MODERATE_CIRCUIT_SPECS = {100, 1000, 200};
    constexpr CircuitSpecs BIG_CIRCUIT_SPECS = {250, 30000, 250};
    constexpr CircuitSpecs HUGE_CIRCUIT_SPECS = {
        std::numeric_limits<unsigned int>::max(),
        std::numeric_limits<unsigned int>::max(),
        std::numeric_limits<unsigned int>::max()
    };

    const std::unordered_map<int, CircuitSpecs>
    CIRCUIT_CLASS_TO_SPECS = {
        {TINY, TINY_CIRCUIT_SPECS},
        {SMALL, SMALL_CIRCUIT_SPECS},
        {MODERATE, MODERATE_CIRCUIT_SPECS},
        {BIG, BIG_CIRCUIT_SPECS},
        {HUGE, HUGE_CIRCUIT_SPECS}
    };

    /**
     *
     * @param nr_qubits
     * @param nr_instructions
     * @param depth
     * @return
     */
    unsigned int classify_circuit(
        unsigned int nr_qubits,
        unsigned int nr_instructions,
        unsigned int depth);
} // namespace ai_pass_selector
#endif // CIRCUIT_UTILS_HPP
