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
    unsigned int index, const fs::path &circuit_path) {
  return environments[index].register_quantum_circuit(circuit_path);
}


void ParallelEnvironments::clear_circuits() {
  for (QuantumCircuitEnviorment &environment : environments) {
    environment.clear_circuit();
  }
}

unsigned int ParallelEnvironments::size() const {
  return static_cast<unsigned int>(environments.size());
}


std::vector<std::tuple<double, bool> > ParallelEnvironments::step(
    std::vector<unsigned int> actions) {
if ()
}

torch::Tensor
ParallelEnvironments::get_batched_instruction_based_observations() const {
  const int64_t B = nr_environments;
  const int64_t H = max_instructions;
  const int64_t W = max_qubits + NR_GATES + MAX_GATE_PARAMS;

  torch::Tensor out = torch::zeros({B, H, W}, torch::kFloat64);
  for (int64_t i = 0; i < B; ++i) {
    auto &env = const_cast<QuantumCircuitEnviorment &>(
      environments[static_cast<size_t>(i)]);
    auto obs = env.get_instruction_based_observation();
    torch::Tensor src = torch::from_blob(
        obs.raw(), {H, W}, torch::kFloat64);
    out.index_put_({i}, src.clone());
  }
  return out;
}

torch::Tensor
ParallelEnvironments::get_batched_depth_based_observations() const {
  const int64_t B = nr_environments;
  const int64_t D = max_depth;
  const int64_t Q = max_qubits;
  const int64_t F = NR_GATES + MAX_GATE_PARAMS + QUBIT_ROLE + max_qubits;

  torch::Tensor out = torch::zeros({B, D, Q, F}, torch::kFloat64);
  for (int64_t i = 0; i < B; ++i) {
    auto &env = const_cast<QuantumCircuitEnviorment &>(
      environments[static_cast<size_t>(i)]);
    auto obs = env.get_depth_based_observation();
    torch::Tensor src = torch::from_blob(
        obs.raw(), {D, Q, F}, torch::kFloat64);
    out.index_put_({i}, src.clone());
  }
  return out;
}


} // namespace ai_pass_selector