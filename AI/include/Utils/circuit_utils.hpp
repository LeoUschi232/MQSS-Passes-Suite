#ifndef CIRCUIT_UTILS_HPP
#define CIRCUIT_UTILS_HPP

#include <array>
#include <filesystem>
#include <vector>
#include <string>
#include <limits>

namespace ai_pass_selector {
/// Circuit specifications [max_qubits, max_instructions, max_depth]
constexpr std::array<unsigned int, 3> TINY_CIRCUIT_SPECS = {5, 25, 10};
constexpr std::array<unsigned int, 3> SMALL_CIRCUIT_SPECS = {20, 500, 50};
constexpr std::array<unsigned int, 3> MODERATE_CIRCUIT_SPECS = {100, 1000, 200};
constexpr std::array<unsigned int, 3> BIG_CIRCUIT_SPECS = {250, 30000, 250};
constexpr std::array<unsigned int, 3> HUGE_CIRCUIT_SPECS = {
    std::numeric_limits<unsigned int>::max(),
    std::numeric_limits<unsigned int>::max(),
    std::numeric_limits<unsigned int>::max()
};

/// Circuit classifiers
constexpr unsigned int TINY = 1;
constexpr unsigned int SMALL = 2;
constexpr unsigned int MODERATE = 3;
constexpr unsigned int BIG = 4;
constexpr unsigned int HUGE = 5;

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

/**
 *
 * @param s
 * @param delim
 * @return
 */
std::vector<std::string> split_string(const std::string &s, char delim);
} // namespace ai_pass_selector
#endif // CIRCUIT_UTILS_HPP
