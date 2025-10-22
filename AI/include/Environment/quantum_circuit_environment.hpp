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

// Torch includes
#include "torch/torch.h"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"

// Utils includes
#include "Utils/info_utils.hpp"

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
constexpr unsigned int MIN_NR_STEPS = 1u;

enum class CircuitValidity : int {
  Valid = 0,
  NoCircuit = 1,
  InvalidNrQubits = 2,
  InvalidNrGates = 3,
  InvalidNrAllocations = 4,
  AmbiguousMeasurement = 5
};

class QuantumCircuitEnvironment {
protected:
  /// Attributes for circuit
  std::optional<InstructionsTensor<float>> latest_observation = std::nullopt;
  unsigned int max_qubits = GLOBAL_MIN_NR_QUBITS;
  fs::path circuit_path = "";
  QuantumCircuit circuit{};
  double nr_gates_reduction_weight = 1.0;

  /// Attributes for episode
  unsigned int max_steps_per_episode = MIN_NR_STEPS;
  unsigned int max_steps_no_improvement = MIN_NR_STEPS;
  unsigned int max_steps_no_change = MIN_NR_STEPS;
  unsigned int max_steps_same_action = MIN_NR_STEPS;
  unsigned int step_per_episode = 0u;
  unsigned int step_no_improvement = 0u;
  unsigned int step_no_change = 0u;
  unsigned int step_same_action = 0u;
  int last_action = -1;
  bool terminated = false;
  bool truncated = false;

  /// Attributes for randomizer
  std::optional<std::array<double, CHOLESKY_PARAMS_SIZE>>
      qubits_cholesky_params = std::nullopt;
  std::optional<std::array<unsigned int, GATES_WEIGHTS_SIZE>> gates_weights =
      std::nullopt;

  /// Other attributes
  torch::Device device = torch::kCPU;

public:
  /// Constructors
  explicit QuantumCircuitEnvironment(unsigned int max_qubits);

  /// Destructor
  virtual ~QuantumCircuitEnvironment() = default;

  /// Copy constructors
  // Forbid copying the QuantumCircuitEnvironment because the MLIRContext is
  // tied exactly to the circuit module and it is ambiguous if you copy both of
  // them if the copies are then untied from their originals but tied to each
  // other.
  QuantumCircuitEnvironment(const QuantumCircuitEnvironment &other) = delete;
  QuantumCircuitEnvironment &
  operator=(const QuantumCircuitEnvironment &other) = delete;

  /// Move Constructors
  QuantumCircuitEnvironment(QuantumCircuitEnvironment &&other) noexcept =
      default;
  QuantumCircuitEnvironment &
  operator=(QuantumCircuitEnvironment &&) noexcept = default;

  /// Getters
  unsigned int getMaxQubits() const;
  fs::path getCircuitPath() const;
  std::optional<std::pair<std::array<double, 11>, std::array<unsigned, 37>>>
  getRegisteredRandomizerParams() const;

  /// Short functions
  void clear(bool hard = true);
  bool validate();
  virtual void reset();
  std::pair<unsigned int, unsigned int> size() const;

  /**
   *
   * @return
   */
  CircuitValidity get_advanced_circuit_validity();

  /**
   *
   * @param circuit_path
   */
  bool register_quantum_circuit(const fs::path &circuit_path);

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
  std::unordered_map<std::string, unsigned int> get_circuit_info() const;

  /**
   * N = Nr of instructions in the quantum circuit
   * IRS = Instruction Representation Size
   * @return Blob Tensor of 1-axis shape {N×IRS} containing the observation of
   * the current circuit.
   */
  InstructionsTensor<float> get_observation();

  /**
   * N = Nr of instructions in the quantum circuit
   * IRS = Instruction Representation Size
   * @param tensor_options
   * @return Torch Tensor of 1-axis shape [N, IRS] containing the observation of
   * the current circuit.
   */
  torch::Tensor get_observation_as_torch_tensor(
      std::optional<torch::TensorOptions> tensor_options = std::nullopt);

  /**
   *
   * @param action
   * @return [Reward, Terminated, Truncated]
   */
  virtual std::tuple<float, bool, bool> step(unsigned int action);

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
