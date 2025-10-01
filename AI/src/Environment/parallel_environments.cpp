#include "Environment/parallel_environments.hpp"

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

  std::vector<std::future<InstructionsTensor<double>>> observation_futures;
  observation_futures.reserve(B);
  for (int64_t i = 0; i < B; i++) {
    observation_futures.emplace_back(std::async(std::launch::async, [&, i] {
      auto &environment =
          const_cast<QuantumCircuitEnvironment &>(environments[i]);
      return environment.get_observation();
    }));
  }
  std::vector<InstructionsTensor<double>> observations;
  observations.reserve(B);
  int64_t maxN = 0u;
  int64_t IRP = 0u;
  for (auto &observation_future : observation_futures) {
    observations.emplace_back(observation_future.get());
    maxN = std::max(static_cast<unsigned>(maxN), observations.back().shape[0]);
    if (IRP <= 0) {
      IRP = observations.back().shape[1];
    } else if (IRP != observations.back().shape[1]) {
      throw std::runtime_error("Inconsistent Instruction Representation Size.");
    }
  }
  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
  if (maxN <= 0) {
    return torch::zeros({B, 1, IRP}, options);
  }

  std::vector<torch::Tensor> torch_tensors;
  torch_tensors.reserve(B);
  for (int64_t i = 0; i < B; ++i) {
    InstructionsTensor<double> instruction_tensor = observations[i];
    instruction_tensor.pad(maxN, 0.0);
    torch::Tensor tensor =
        torch::from_blob(instruction_tensor.raw(), {maxN, IRP}, options)
            .clone();
    torch_tensors.emplace_back(std::move(tensor));
  }
  return torch::stack(torch_tensors, 0);
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