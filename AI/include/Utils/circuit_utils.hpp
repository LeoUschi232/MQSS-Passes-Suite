#ifndef CIRCUIT_UTILS_HPP
#define CIRCUIT_UTILS_HPP

// Standard Library includes
#include <limits>
#include <string>
#include <unordered_map>

namespace ai_pass_selector {
/// Circuit size classes
constexpr int TINY = 1;
constexpr int SMALL = 2;
constexpr int MODERATE = 3;
constexpr int BIG = 4;
constexpr int HUGE = 5;

const std::unordered_map<std::string, int> CIRCUIT_SIZE_NAME_TO_CLASS = {
    {"tiny", TINY},
    {"small", SMALL},
    {"moderate", MODERATE},
    {"big", BIG},
    {"huge", HUGE}};
const std::unordered_map<int, std::string> CIRCUIT_SIZE_CLASS_TO_NAME = {
    {TINY, "tiny"},
    {SMALL, "small"},
    {MODERATE, "moderate"},
    {BIG, "big"},
    {HUGE, "huge"}};

constexpr unsigned int TINY_CIRCUIT_MAX_QUBITS = 5;
constexpr unsigned int SMALL_CIRCUIT_MAX_QUBITS = 20;
constexpr unsigned int MODERATE_CIRCUIT_MAX_QUBITS = 100;
constexpr unsigned int BIG_CIRCUIT_MAX_QUBITS = 250;
constexpr unsigned int HUGE_CIRCUIT_MAX_QUBITS =
    std::numeric_limits<unsigned int>::max();

const std::unordered_map<int, unsigned int> CIRCUIT_SIZE_CLASS_TO_MAX_QUBITS = {
    {TINY, TINY_CIRCUIT_MAX_QUBITS},
    {SMALL, SMALL_CIRCUIT_MAX_QUBITS},
    {MODERATE, MODERATE_CIRCUIT_MAX_QUBITS},
    {BIG, BIG_CIRCUIT_MAX_QUBITS},
    {HUGE, HUGE_CIRCUIT_MAX_QUBITS}};

/**
 *
 * @param nr_qubits
 * @return
 */
unsigned int classify_circuit(unsigned int nr_qubits);
} // namespace ai_pass_selector
#endif // CIRCUIT_UTILS_HPP
