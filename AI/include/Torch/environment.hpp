#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

// MLIR includes
#include "mlir/Pass/Pass.h"

// Torch includes
#include <torch/torch.h>

// Standard library includes
#include <unordered_map>
#include <utility>


namespace ai_pass_selector {
class QuantumCircuitEnviorment {
  int max_qubits;
  int max_instructions;
  int max_depth;

public:
  /// Constructor
  QuantumCircuitEnviorment(int max_qubits, int max_instructions, int max_depth);

  /// Destructor
  ~QuantumCircuitEnviorment() = default;

  /// Copy and move constructors and assignment operators
  QuantumCircuitEnviorment(const QuantumCircuitEnviorment &other) = delete;

  QuantumCircuitEnviorment(QuantumCircuitEnviorment &&other) noexcept = default;

  QuantumCircuitEnviorment
  &operator=(const QuantumCircuitEnviorment &other) = delete;

  QuantumCircuitEnviorment
  &operator=(const QuantumCircuitEnviorment &&other) noexcept = delete;

  /**
   *
   * @param seed
   * @return
   */
  std::pair<torch::Tensor, std::unordered_map<std::string, int> >
  reset(int seed = 0);

  bool is_valid_circuit();

  /**
   *
   * @return
   */
  std::unordered_map<std::string, int> get_circuit_info() const;

};

} // namespace ai_pass_selector

#endif // ENVIRONMENT_HPP