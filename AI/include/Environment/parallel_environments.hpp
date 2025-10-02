#ifndef PARALLEL_ENVIRONMENTS_HPP
#define PARALLEL_ENVIRONMENTS_HPP

// Environment includes
#include "Environment/environment.hpp"

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
  ParallelEnvironments(unsigned int nr_environments, unsigned int max_qubits,
                       unsigned int max_steps);

  /// Destructor
  ~ParallelEnvironments() = default;

  /// Copy and move constructors and assignment operators
  ParallelEnvironments(const ParallelEnvironments &other) = delete;

  ParallelEnvironments &operator=(const ParallelEnvironments &other) = delete;

  ParallelEnvironments(ParallelEnvironments &&other) noexcept = default;

  ParallelEnvironments &operator=(ParallelEnvironments &&) noexcept = default;

  /**
   *
   * @param index Environment slot that should own the circuit.
   * @param circuit_path Path to the circuit that should be registered.
   * @return True if the circuit could be registered successfully.
   */
  bool register_quantum_circuit(unsigned int index,
                                const fs::path &circuit_path);

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
   * IRP = Instruction representation size
   * @return Torch tensor of shape {B, N, IRP} containing the batched
   * observations of all environments and mask of shape {B, N} containing 1.0 if
   * legit instruction and 0.0 if padding.
   */
  torch::Tensor get_batched_observations() const;

  /**
   *
   * @param compute_padding
   * @return
   */
  std::pair<torch::Tensor, unsigned int>
  get_batched_observations_with_padding(bool compute_padding = true) const;

  /**
   *
   * @param actions
   * @return
   */
  std::tuple<std::vector<double>, std::vector<bool>>
  step(const std::vector<unsigned int> &actions);

  /// Small methods
  unsigned int size() const;
  void clear();
  void reset();
};
} // namespace ai_pass_selector

#endif // PARALLEL_ENVIRONMENTS_HPP
