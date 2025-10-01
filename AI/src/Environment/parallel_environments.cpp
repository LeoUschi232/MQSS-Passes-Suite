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
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
    const std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights) {
  assert(environments.size() == nr_environments &&
         "environments.size() != nr_environments");
  this->cholesky_params = cholesky_params;
  this->gates_weights = gates_weights;
  for (QuantumCircuitEnvironment &environment : environments) {
    environment.register_randomizer_params(cholesky_params, gates_weights);
  }
}

bool ParallelEnvironments::randomize_all_circuits_with_equal_dimensions() {
  if (this->environments.size() != this->nr_environments ||
      this->nr_environments <= 0 || !this->cholesky_params.has_value() ||
      !this->gates_weights.has_value()) {
    return false;
  }
  auto [nr_qubits, nr_gates, nr_operations, nr_measurements] =
      sample_nr_qubits_gates_operations_measurements(cholesky_params.value());

  return true;
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
  for (int64_t b = 0; b < B; b++) {
    InstructionsTensor<double> instruction_tensor = observations[b];
    instruction_tensor.pad(maxN, 0.0);
    torch::Tensor tensor =
        torch::from_blob(instruction_tensor.raw(), {maxN, IRP}, options)
            .clone();
    torch_tensors.emplace_back(std::move(tensor));
  }
  return torch::stack(torch_tensors);
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