#include "Torch/parallel_environments.hpp"

#include <future>

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
  if (index >= nr_environments) {
    throw std::out_of_range("In register_quantum_circuit.");
  }
  return environments[index].register_quantum_circuit(circuit_path);
}


void ParallelEnvironments::clear_circuits() {
  for (QuantumCircuitEnviorment &environment : environments) {
    environment.clear_circuit();
  }
}

unsigned int ParallelEnvironments::size() const {
  return nr_environments;
}


std::tuple<std::vector<double>, std::vector<bool> >
ParallelEnvironments::step(const std::vector<unsigned int> &actions) {
  if (actions.size() != nr_environments) {
    throw std::runtime_error("actions.size() != nr_environments");
  }
  std::vector<std::future<std::tuple<double, bool> > > futures;
  futures.reserve(nr_environments);
  for (size_t i = 0; i < nr_environments; ++i) {
    futures.emplace_back(std::async(std::launch::async, [&, i] {
      return environments[i].step(actions[i]);
    }));
  }
  std::vector<double> rewards;
  std::vector<bool> terminates;
  rewards.reserve(nr_environments);
  terminates.reserve(nr_environments);
  for (auto &future : futures) {
    auto result = future.get();
    rewards.push_back(std::get<0>(result));
    terminates.push_back(std::get<1>(result));
  }
  return {std::move(rewards), std::move(terminates)};
}


torch::Tensor
ParallelEnvironments::get_batched_instruction_based_observations() const {
  const int64_t B = nr_environments, H = max_instructions;
  const int64_t W = max_qubits + NR_GATES + MAX_GATE_PARAMS;

  std::vector<std::future<torch::Tensor> > futures;
  futures.reserve(B);
  for (int64_t i = 0; i < B; ++i) {
    futures.emplace_back(std::async(std::launch::async, [&, i] {
      auto &env = const_cast<QuantumCircuitEnviorment &>(environments[i]);
      auto obs = env.get_instruction_based_observation();
      auto src = torch::from_blob(obs.raw(), {H, W}, torch::kFloat64);
      return src.clone();
    }));
  }
  std::vector<torch::Tensor> slices;
  slices.reserve(B);
  for (auto &future : futures) {
    slices.emplace_back(future.get());
  }
  return torch::stack(slices, 0);
}

torch::Tensor
ParallelEnvironments::get_batched_depth_based_observations() const {
  const int64_t B = nr_environments, D = max_depth, Q = max_qubits;
  const int64_t F = NR_GATES + MAX_GATE_PARAMS + QUBIT_ROLE + max_qubits;

  std::vector<std::future<torch::Tensor> > futures;
  futures.reserve(B);
  for (int64_t i = 0; i < B; ++i) {
    futures.emplace_back(std::async(std::launch::async, [&, i] {
      auto &env = const_cast<QuantumCircuitEnviorment &>(environments[i]);
      auto obs = env.get_depth_based_observation();
      auto src = torch::from_blob(obs.raw(), {D, Q, F}, torch::kFloat64);
      return src.clone();
    }));
  }
  std::vector<torch::Tensor> slices;
  slices.reserve(B);
  for (auto &future : futures) {
    slices.emplace_back(future.get());
  }
  return torch::stack(slices, 0);
}


} // namespace ai_pass_selector