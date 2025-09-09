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
#include "mlir/Pass/Pass.h"

// Support includes
#include "Support/CodeGen/Quake.hpp"

// Standard library includes
#include <string>
#include <unordered_map>
#include <utility>

using namespace mqss::support::quakeDialect;


namespace ai_pass_selector {


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
   * @param seed
   * @return
   */
  std::pair<QuantumCircuitTensor<double>, std::unordered_map<std::string, int> >
  reset(int seed = 0);

  /**
   *
   * @param circuit
   * @return
   */
  static std::unordered_map<std::string, int>
  get_circuit_info(ModuleOp circuit);


  /**
   *
   * @return
   */
  std::unordered_map<std::string, int> get_circuit_info() const;

  /**
   *
   * @return
   */
  bool is_valid_circuit() const;

  /**
   *
   * @return
   */
  QuantumCircuitTensor<double> get_observation();


};

} // namespace ai_pass_selector

#endif // ENVIRONMENT_HPP