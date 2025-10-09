#ifndef PARALLEL_ENVIRONMENTS_HPP
#define PARALLEL_ENVIRONMENTS_HPP

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
class ParallelEnvironments {
  /// Attributes for environments
  unsigned int nr_environments;
  std::vector<QuantumCircuitEnvironment> environments;
  unsigned int max_qubits;
  unsigned int max_steps;

  /// Attributes for randomizer
  std::optional<std::array<double, CHOLESKY_PARAMS_SIZE>>
      qubits_cholesky_params;
  std::optional<std::array<unsigned int, GATES_WEIGHTS_SIZE>> gates_weights;

public:
  /// Constructor
  ParallelEnvironments(unsigned int max_qubits, unsigned int max_steps,
                       unsigned int nr_environments = 1u);

  /// Destructor
  ~ParallelEnvironments() = default;

  /// Copy and move constructors and assignment operators
  ParallelEnvironments(const ParallelEnvironments &other) = delete;

  ParallelEnvironments &operator=(const ParallelEnvironments &other) = delete;

  ParallelEnvironments(ParallelEnvironments &&other) noexcept = default;

  ParallelEnvironments &operator=(ParallelEnvironments &&) noexcept = default;

  /**
   *
   * @param circuit_path Path to the circuit that should be registered.
   * @param index Environment slot that should own the circuit.
   * @return True if the circuit could be registered successfully.
   */
  bool register_quantum_circuit(const fs::path &circuit_path,
                                unsigned int index = 0u);

  /**
   *
   * @param cholesky_params
   * @param gates_weights
   */
  void register_randomizer_params(
      const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
      const std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights);

  /**
   *
   * @return
   */
  std::tuple<bool, unsigned int, unsigned int>
  randomize_all_circuits_with_equal_dimensions();

  /**
   * B = Batch size / Nr of parallel environments
   * N = Nr of instructions in the quantum circuit
   * IRS = Instruction Representation Size
   * @return Torch tensor of shape [N, IRS] if nr_environments=1, or {B, N, IRS}
   * if nr_environments>=2 containing the batched observations of all
   * environments.
   */
  torch::Tensor get_observation() const;

  /**
   *
   * @return [batched_observations, mask(instr=1.0|padding=0.0)]
   */
  std::pair<torch::Tensor, torch::Tensor>
  get_batched_observations_and_mask() const;

  /**
   *
   * @param actions
   * @return Vector of [Reward, Terminated, Truncated]
   */
  std::vector<std::tuple<double, bool, bool>>
  step(const torch::Tensor &actions);

  /// Small methods
  unsigned int size() const;
  void clear();
  void reset();
};
} // namespace ai_pass_selector

#endif // PARALLEL_ENVIRONMENTS_HPP
