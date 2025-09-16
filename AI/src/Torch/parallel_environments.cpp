#include "Torch/parallel_environments.hpp"

namespace ai_pass_selector {
ParallelEnvironments::ParallelEnvironments(
    unsigned int nr_environments,
    unsigned int max_qubits,
    unsigned int max_instructions,
    unsigned int max_depth,
    unsigned int max_steps)
  : nr_environments(nr_environments),
    max_qubits(max_qubits),
    max_instructions(max_instructions),
    max_depth(max_depth),
    max_steps(max_steps) {
  environments.reserve(nr_environments);
  for (unsigned int i = 0; i < nr_environments; i++) {
    environments.emplace_back(
        max_qubits, max_instructions, max_depth, max_steps);
  }
}

bool ParallelEnvironments::register_quantum_circuit(
    unsigned int index,  const fs::path &circuit_path) {
  return environments[index].register_quantum_circuit(circuit_path);
}

} // namespace ai_pass_selector