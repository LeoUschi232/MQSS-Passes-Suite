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
constexpr unsigned int TOO_MANY_QUBITS = 2;
constexpr unsigned int NO_QUBIT_ALLOCATIONS = 3;
constexpr unsigned int MULTIPLE_QUBIT_ALLOCATIONS = 4;
constexpr unsigned int AMBIGUOUS_MEASUREMENT = 5;

class QuantumCircuitEnviorment {
  unsigned int max_qubits;
  fs::path circuit_path;
  ModuleOp circuit_module;
  std::unique_ptr<MLIRContext *> context_ptr;
  unsigned int max_steps;
  unsigned int current_step;

public:
  /// Constructors
  QuantumCircuitEnviorment(unsigned int max_qubits, unsigned int max_steps,
                           const fs::path &circuit_path = "");

  /// Destructor
  ~QuantumCircuitEnviorment() = default;

  /// Copy and move constructors and assignment operators
  QuantumCircuitEnviorment(const QuantumCircuitEnviorment &other) = delete;

  QuantumCircuitEnviorment(QuantumCircuitEnviorment &&other) noexcept = default;

  QuantumCircuitEnviorment &
  operator=(const QuantumCircuitEnviorment &other) = delete;

  QuantumCircuitEnviorment &
  operator=(QuantumCircuitEnviorment &&) noexcept = default;

  /**
   *
   * @param circuit_path
   */
  bool register_quantum_circuit(const fs::path &circuit_path);

  /**
   *
   */
  void clear_circuit();

  /**
   *
   */
  void reset();

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
   * @return
   */
  std::tuple<double, bool> step(unsigned int action);

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
