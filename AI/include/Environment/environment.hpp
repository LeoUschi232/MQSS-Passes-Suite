#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

////////////////////////////////////////////////////////////////////////////////
/// The includes of llvm Casting must be left here before the include of cudaq
/// QuakeOps otherwise the comipler will complain that these operations do not
/// exist in the header file.
#include "Support/mlir_utils.hpp"

#include "llvm/Support/Casting.h"
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
////////////////////////////////////////////////////////////////////////////////

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"
#include "Environment/random_quantum_circuit_generator.hpp"
#include "Environment/statistics_for_rqcg.hpp"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"

// Standard library includes
#include <filesystem>
#include <string>
#include <unordered_map>

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::ModuleOp;
////////////////////////////////////////////////////////////////////////////////

using namespace mqss::support::quakeDialect;
namespace fs = std::filesystem;

namespace ai_pass_selector {
constexpr unsigned int CIRCUIT_VALID = 0;
constexpr unsigned int NO_CIRCUIT = 1;
constexpr unsigned int INVALID_NR_QUBITS = 2;
constexpr unsigned int INVALID_NR_ALLOCATIONS = 3;
constexpr unsigned int AMBIGUOUS_MEASUREMENT = 4;

class QuantumCircuitEnvironment {
  /// Attributes for circuit
  unsigned int max_qubits;
  ModuleOp circuit_module;
  std::unique_ptr<MLIRContext *> context_ptr;
  fs::path circuit_path = "";
  unsigned int nr_qubits = 0u;
  unsigned int nr_gates = 0u;
  unsigned int depth = 0u;

  /// Attributes for episode
  unsigned int max_steps_per_episode = 1u;
  unsigned int max_steps_no_improvement = 1u;
  unsigned int max_steps_no_change = 1u;
  unsigned int max_steps_same_action = 1u;
  unsigned int step_per_episode = 0u;
  unsigned int step_no_improvement = 0u;
  unsigned int step_no_change = 0u;
  unsigned int step_same_action = 0u;
  int last_action = -1;
  bool terminated = false;
  bool truncated = false;

  /// Attributes for randomizer
  std::optional<std::array<double, CHOLESKY_PARAMS_SIZE>>
      qubits_cholesky_params;
  std::optional<std::array<unsigned int, GATES_WEIGHTS_SIZE>> gates_weights;

public:
  /// Constructors
  QuantumCircuitEnvironment(unsigned int max_qubits, unsigned int max_steps,
                            const fs::path &circuit_path = "");

  QuantumCircuitEnvironment(
      unsigned int max_qubits,
      std::unordered_map<std::string, std::string> params);

  /// Destructor
  ~QuantumCircuitEnvironment() {
    if (this->context_ptr) {
      delete *this->context_ptr.get();
    }
  }

  /// Copy and move constructors and assignment operators
  // Forbid copying the QuantumCircuitEnvironment because the MLIRContext is
  // tied exactly to the circuit module and it is ambiguous if you copy both of
  // them if the copies are then untied from their originals but tied to each
  // other.
  QuantumCircuitEnvironment(const QuantumCircuitEnvironment &other) = delete;

  QuantumCircuitEnvironment &
  operator=(const QuantumCircuitEnvironment &other) = delete;

  QuantumCircuitEnvironment(QuantumCircuitEnvironment &&other) noexcept;

  QuantumCircuitEnvironment &operator=(QuantumCircuitEnvironment &&) noexcept;

  /// Clear and Reset
  void clear(bool hard = true);
  void reset();

  /**
   *
   * @param circuit_path
   */
  bool register_quantum_circuit(const fs::path &circuit_path);

  /**
   *
   * @param cholesky_params
   * @param gates_weights
   * @param randomizer_options
   * @return
   */
  bool custom_randomize_circuit(
      const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
      const std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights,
      const RandomizerOptions &randomizer_options);

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
   * @param circuit
   * @return
   */
  static std::unordered_map<std::string, unsigned int>
  get_circuit_info(FuncOp circuit);

  /**
   *
   * @param circuit
   * @return
   */
  unsigned int circuit_invalid_type(FuncOp circuit) const;

  /**
   *
   * @return
   */
  std::unordered_map<std::string, unsigned int> get_circuit_info() const;

  /**
   * B = Batch size / Nr of parallel environments
   * N = Nr of instructions in the quantum circuit
   * IRP = Instruction representation size
   * The transformation from shape {N×IRP} to {B, N, IRP} will be done by the
   * ParallelEnvironments object.
   * @return Blob tensor of 1-axis shape {N×IRP} containing the observation of
   * the current circuit.
   */
  InstructionsTensor<double> get_observation();

  /**
   *
   * @param action
   * @return [Reward, Terminated, Truncated]
   */
  std::tuple<double, bool, bool> step(unsigned int action);

  /**
   *
   * @param op
   * @return
   */
  static std::tuple<std::vector<int>, std::vector<int>, std::vector<double>,
                    bool>
  getOperatingControlsTargetsParams(Operation *op);
};
} // namespace ai_pass_selector

#endif // ENVIRONMENT_HPP
