#include "../../include/Environment/parallel_environments.hpp"

#include <future>

namespace ai_pass_selector {
ParallelEnvironments::ParallelEnvironments(unsigned int nr_environments,
                                           unsigned int max_qubits,
                                           unsigned int max_steps)
    : nr_environments(nr_environments), max_qubits(max_qubits),
      max_steps(max_steps) {
  environments.reserve(nr_environments);
  for (unsigned int i = 0; i < nr_environments; i++) {
    environments.emplace_back(max_qubits, max_steps);
  }
}

bool ParallelEnvironments::register_quantum_circuit(
    unsigned int index, const fs::path &circuit_path) {
  if (index >= nr_environments) {
    throw std::out_of_range(
        "register_quantum_circuit index >= nr_environments");
  }
  return environments[index].register_quantum_circuit(circuit_path);
}

void ParallelEnvironments::register_randomizer_params(
    unsigned int index,
    const std::tuple<double, double, double, double, double>
        &qubits_and_gates_distribution_params,
    const std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights) {
  if (index >= nr_environments) {
    throw std::out_of_range(
        "register_randomizer_params index >= nr_environments");
  }
  environments[index].register_randomizer_params(
      qubits_and_gates_distribution_params, gates_weights);
}

void ParallelEnvironments::register_randomizer_params(
    const std::tuple<double, double, double, double, double>
        &qubits_and_gates_distribution_params,
    const std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights) {
  assert(environments.size() == nr_environments &&
         "environments.size() != nr_environments");
  for (QuantumCircuitEnvironment &environment : environments) {
    environment.register_randomizer_params(qubits_and_gates_distribution_params,
                                           gates_weights);
  }
}

std::tuple<std::vector<double>, std::vector<bool>>
ParallelEnvironments::step(const std::vector<unsigned int> &actions) {
  if (actions.size() != nr_environments) {
    throw std::runtime_error("actions.size() != nr_environments");
  }
  std::vector<std::future<std::tuple<double, bool>>> futures;
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
    auto [reward, terminate] = future.get();
    rewards.push_back(reward);
    terminates.push_back(terminate);
  }
  return {std::move(rewards), std::move(terminates)};
}

torch::Tensor ParallelEnvironments::get_batched_observations() const {
  const int64_t B = nr_environments;

  std::vector<std::future<torch::Tensor>> futures;
  futures.reserve(B);
  for (int64_t i = 0; i < B; ++i) {
    futures.emplace_back(std::async(std::launch::async, [&, i] {
      auto &env = const_cast<QuantumCircuitEnvironment &>(environments[i]);
      auto obs = env.get_observation();
      // N = Nr of instructions in the quantum circuit
      // IRP = Instruction representation size
      const int64_t N = obs.shape[0];
      const int64_t IRP = obs.shape[1];
      auto src = torch::from_blob(obs.raw(), {N, IRP}, torch::kFloat64);
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

unsigned int ParallelEnvironments::size() const { return nr_environments; }
void ParallelEnvironments::clear() {
  for (QuantumCircuitEnvironment &environment : environments) {
    environment.clear();
  }
}
void ParallelEnvironments::reset() {
  for (QuantumCircuitEnvironment &environment : environments) {
    environment.reset();
  }
}
} // namespace ai_pass_selector