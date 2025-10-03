#ifndef QUANTUM_CIRCUIT_HPP
#define QUANTUM_CIRCUIT_HPP

// MLIR includes
#include "mlir/IR/BuiltinOps.h"

// Standard library includes
#include <filesystem>

namespace fs = std::filesystem;
using mlir::MLIRContext;
using mlir::ModuleOp;

namespace ai_pass_selector {
class QuantumCircuit {
  ModuleOp circuit_module = nullptr;
  std::unique_ptr<MLIRContext> context_ptr = nullptr;
  unsigned int nr_qubits = 0u;
  unsigned int nr_gates = 0u;
  unsigned int depth = 0u;

public:
  /// Constructors
  explicit QuantumCircuit(const fs::path &circuit_path = "") {
    this->set(circuit_path);
  }
  QuantumCircuit(ModuleOp circuit_module,
                 std::unique_ptr<MLIRContext> context_ptr) {
    this->set(circuit_module, std::move(context_ptr));
  }
  QuantumCircuit(ModuleOp circuit_module,
                 std::unique_ptr<MLIRContext> context_ptr,
                 unsigned int nr_qubits, unsigned int nr_gates,
                 unsigned int depth) {
    this->set(circuit_module, std::move(context_ptr), nr_qubits, nr_gates,
              depth);
  }

  /// Destructor
  ~QuantumCircuit() = default;

  /// Setters
  bool set(const fs::path &circuit_path = "");
  bool set(ModuleOp circuit_module, std::unique_ptr<MLIRContext> context_ptr);
  bool set(ModuleOp circuit_module, std::unique_ptr<MLIRContext> context_ptr,
           unsigned int nr_qubits, unsigned int nr_gates, unsigned int depth);

  /// Clear nad validate
  void clear();
  bool validate();
  bool recompute();

  /**
   *
   * @param pass_index
   * @return
   */
  bool run_pass(unsigned int pass_index);
};
} // namespace ai_pass_selector

#endif // QUANTUM_CIRCUIT_HPP