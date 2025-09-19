#include "Utils/circuit_utils.hpp"

#include <filesystem>

namespace fs = std::filesystem;

namespace ai_pass_selector {


unsigned int classify_circuit(
    unsigned int nr_qubits,
    unsigned int nr_instructions,
    unsigned int depth) {
  auto [max_qubits, max_instructions, max_depth] = TINY_CIRCUIT_SPECS;
  if (nr_qubits <= max_qubits
      && nr_instructions <= max_instructions
      && depth <= max_depth) {
    return TINY;
  }
  std::tie(max_qubits, max_instructions, max_depth) = SMALL_CIRCUIT_SPECS;
  if (nr_qubits <= max_qubits
      && nr_instructions <= max_instructions
      && depth <= max_depth) {
    return SMALL;
  }
  std::tie(max_qubits, max_instructions, max_depth) = MODERATE_CIRCUIT_SPECS;
  if (nr_qubits <= max_qubits
      && nr_instructions <= max_instructions
      && depth <= max_depth) {
    return MODERATE;
  }
  std::tie(max_qubits, max_instructions, max_depth) = BIG_CIRCUIT_SPECS;
  if (nr_qubits <= max_qubits
      && nr_instructions <= max_instructions
      && depth <= max_depth) {
    return BIG;
  }
  return HUGE;
}


} // namespace ai_pass_selector