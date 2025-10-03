#include "Environment/parallel_environments.hpp"

// Utils includes
#include "Utils/info_utils.hpp"

// Standard library includes
#include <future>

namespace ai_pass_selector {
ParallelEnvironments::ParallelEnvironments(unsigned int nr_environments,
                                           unsigned int max_qubits,
                                           unsigned int max_steps)
    : nr_environments(std::max(1u, nr_environments)),
      max_qubits(std::max(GLOBAL_MIN_NR_QUBITS, max_qubits)),
      max_steps(std::max(1u, max_steps)) {
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
  this->qubits_cholesky_params = cholesky_params;
  this->gates_weights = gates_weights;
  for (QuantumCircuitEnvironment &environment : environments) {
    environment.register_randomizer_params(cholesky_params, gates_weights);
  }
}

std::tuple<bool, unsigned int, unsigned int>
ParallelEnvironments::randomize_all_circuits_with_equal_dimensions() {
  if (this->environments.size() != this->nr_environments ||
      this->nr_environments <= 0 || !this->qubits_cholesky_params.has_value() ||
      !this->gates_weights.has_value()) {
    return {false, 0u, 0u};
  }
  try {
    auto [nr_qubits, nr_gates, _, __] =
        sample_nr_qubits_gates_operations_measurements(
            qubits_cholesky_params.value());
    // Ignore nr_measurements because after if nr_oprations is set to
    // nr_gates-nr_qubits, the random circuit generator will infer
    // nr_measurements=nr_gates-nr_operations=nr_qubits.
    // This will create a circuit that measures all qubits at the end.
    nr_qubits =
        std::max(GLOBAL_MIN_NR_QUBITS, std::min(nr_qubits, this->max_qubits));
    nr_gates = std::max(nr_qubits + GLOBAL_MIN_NR_GATES, nr_gates);

    // Technically allow_measurements_as_gates is false by default but I do not
    // and may not ever trust the C++ compiler.
    RandomizerOptions randomizer_options = {
        .exact_nr_qubits = static_cast<int>(nr_qubits),
        .exact_nr_gates = static_cast<int>(nr_gates),
        .exact_nr_operations = static_cast<int>(nr_gates - nr_qubits),
        .allow_measurements_as_gates = false,
        .weight_min_multiplier_for_unoccurring_gates = 0.1,
        .probability_additionals_controls = 0.01};

    std::vector<std::future<bool>> environment_futures;
    environment_futures.reserve(this->nr_environments);
    for (unsigned int i = 0; i < this->nr_environments; i++) {
      environment_futures.emplace_back(std::async(std::launch::async, [&, i] {
        return environments[i].custom_randomize_circuit(
            qubits_cholesky_params.value(), gates_weights.value(),
            randomizer_options);
      }));
    }
    bool success = true;
    for (auto &environment_future : environment_futures) {
      success &= environment_future.get();
    }
    return {success, nr_qubits, nr_gates};
  } catch (const std::runtime_error &error) {
    std::cerr << error.what() << std::endl;
  }
  return {false, 0u, 0u};
}

std::vector<std::tuple<double, bool, bool>>
ParallelEnvironments::step(const torch::Tensor &actions) {
  if (actions.size(/*dim=*/0) != nr_environments) {
    throw std::runtime_error("actions.size() != nr_environments");
  }
  std::vector<std::future<std::tuple<double, bool, bool>>> futures;
  futures.reserve(nr_environments);
  for (size_t i = 0; i < nr_environments; ++i) {
    futures.emplace_back(std::async(std::launch::async, [&, i] {
      return environments[i].step(actions[i].item<int>());
    }));
  }
  std::vector<std::tuple<double, bool, bool>> step_returns;
  step_returns.reserve(nr_environments);
  for (auto &future : futures) {
    step_returns.push_back(future.get());
  }
  return step_returns;
}

torch::Tensor ParallelEnvironments::get_batched_observations() const {
  auto [batched_observations, _] = get_batched_observations_with_padding();
  return batched_observations;
}

std::pair<torch::Tensor, torch::Tensor>
ParallelEnvironments::get_batched_observations_with_padding() const {
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

  unsigned int maxN = 0u;
  int64_t IRP = 0u;
  for (auto &observation_future : observation_futures) {
    observations.emplace_back(observation_future.get());
    maxN = std::max(maxN, observations.back().shape[0]);
    if (IRP <= 0) {
      IRP = observations.back().shape[1];
    } else if (IRP != observations.back().shape[1]) {
      throw std::runtime_error("Inconsistent Instruction Representation Size.");
    }
  }
  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
  if (maxN <= 0) {
    return {torch::zeros({B, 1, IRP}, options), torch::ones({B, 1}, options)};
  }

  std::vector<torch::Tensor> torch_tensors;
  torch_tensors.reserve(B);
  torch::Tensor instruction_mask = torch::ones({B, maxN}, options);
  for (int64_t batch = 0; batch < B; batch++) {
    InstructionsTensor<double> instruction_tensor = observations[batch];
    unsigned int N = instruction_tensor.shape[0];
    instruction_tensor.pad(maxN, 0.0);
    torch::Tensor tensor =
        torch::from_blob(instruction_tensor.raw(), {maxN, IRP}, options)
            .clone();
    torch_tensors.emplace_back(std::move(tensor));
    if (N < maxN) {
      instruction_mask[batch].slice(/*dim=*/0, /*start=*/N, /*end=*/maxN) =
          torch::zeros({static_cast<int64_t>(maxN - N)}, options);
    }
  }
  return {torch::stack(torch_tensors), instruction_mask};
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