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

//  Quake includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// Torch includes
#include <torch/torch.h>

// Standard library includes
#include <string>
#include <unordered_map>
#include <utility>


namespace ai_pass_selector {
class QuantumCircuitEnviorment {
  int max_qubits;
  int max_instructions;
  int max_depth;
  mlir::ModuleOp original_circuit;
  mlir::ModuleOp current_circuit;

public:
  /// Constructor
  QuantumCircuitEnviorment(
      int max_qubits, int max_instructions, int max_depth,
      mlir::ModuleOp circuit);

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

  static int printOperation(mlir::Operation *op);

  /**
   *
   * @return
   */
  bool is_valid_circuit();

  /**
   *
   * @return
   */
  std::unordered_map<std::string, int> get_circuit_info() const;


};

} // namespace ai_pass_selector

#endif // ENVIRONMENT_HPP