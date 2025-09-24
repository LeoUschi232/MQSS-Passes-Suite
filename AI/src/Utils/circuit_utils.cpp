#include "Utils/circuit_utils.hpp"

#include <filesystem>

namespace fs = std::filesystem;

namespace ai_pass_selector {

unsigned int classify_circuit(unsigned int nr_qubits) {
  if (nr_qubits <= TINY_CIRCUIT_MAX_QUBITS) {
    return TINY;
  }
  if (nr_qubits <= SMALL_CIRCUIT_MAX_QUBITS) {
    return SMALL;
  }
  if (nr_qubits <= MODERATE_CIRCUIT_MAX_QUBITS) {
    return MODERATE;
  }
  if (nr_qubits <= BIG_CIRCUIT_MAX_QUBITS) {
    return BIG;
  }
  return HUGE;
}

} // namespace ai_pass_selector