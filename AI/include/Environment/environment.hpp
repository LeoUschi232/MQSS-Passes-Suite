#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

////////////////////////////////////////////////////////////////////////////////
/// The includes of llvm Casting must be left here before the include of cudaq
/// QuakeOps otherwise the comipler will complain that these operations do not
/// exist in the header file.
#include "llvm/Support/Casting.h"
using llvm::isa;
using llvm::cast;
using llvm::dyn_cast;
////////////////////////////////////////////////////////////////////////////////

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"

// Support includes
#include "Support/CodeGen/Quake.hpp"

// Standard library includes
#include <string>
#include <unordered_map>

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::ModuleOp;
////////////////////////////////////////////////////////////////////////////////

using namespace mqss::support::quakeDialect;


namespace ai_pass_selector {
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 6.28318530717958647692;

constexpr int CIRCUIT_VALID = 0;
constexpr int NO_CIRCUIT = 1;
constexpr int TOO_MANY_QUBITS = 2;
constexpr int TOO_MANY_INSTRUCTIONS = 3;
constexpr int TOO_LARGE_DEPTH = 4;
constexpr int NO_QUBIT_ALLOCATIONS = 5;
constexpr int MULTIPLE_QUBIT_ALLOCATIONS = 6;
constexpr int AMBIGUOUS_MEASUREMENT = 7;

inline std::vector<double> params_to_angles(std::vector<double> params) {
  for (int i = 0; i < params.size(); i++) {
    double angle = std::fmod(params[i] + PI, TWO_PI);
    if (angle < 0) {
      angle += TWO_PI;
    }
    params[i] = angle - PI;
  }
  return params;
}

class QuantumCircuitEnviorment {
  int max_qubits;
  int max_instructions;
  int max_depth;
  ModuleOp original_circuit;
  ModuleOp current_circuit;

public:
  /// Constructor
  QuantumCircuitEnviorment(
      int max_qubits, int max_instructions, int max_depth,
      ModuleOp circuit);

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
   * @param circuit
   */
  void register_quantum_circuit(ModuleOp circuit);

  /**
   *
   * @return
   */
  std::tuple<InstructionBasedTensor<double>, DepthBasedTensor<double>,
             std::unordered_map<std::string, int> > reset();

  /**
   *
   * @param circuit
   * @return
   */
  static std::unordered_map<std::string, int>
  get_circuit_info(const ModuleOp &circuit);

  /**
   *
   * @param circuit
   * @return
   */
  int circuit_invalid_type(ModuleOp circuit) const;


  /**
   *
   * @return
   */
  std::unordered_map<std::string, int> get_circuit_info() const;

  /**
   *
   * @return
   */
  InstructionBasedTensor<double> get_instruction_based_observation();

  /**
   *
   * @return
   */
  DepthBasedTensor<double> get_depth_based_observation();


  /**
   *
   * @param op
   * @return
   */
  static std::tuple<std::vector<int>, std::vector<int>, std::vector<double> >
  getOperatingControlsTargetsParams(Operation *op);

};

} // namespace ai_pass_selector

#endif // ENVIRONMENT_HPP