#ifndef QUANTUM_CIRCUIT_HPP
#define QUANTUM_CIRCUIT_HPP

// MLIR includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

// Standard library includes
#include <filesystem>

namespace fs = std::filesystem;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::func::FuncOp;

namespace ai_pass_selector {
class QuantumCircuit {
  ModuleOp circuit_module = nullptr;
  std::unique_ptr<MLIRContext> context_ptr = nullptr;
  unsigned int nr_qubits = 0u;
  unsigned int nr_gates = 0u;
  unsigned int depth = 0u;

public:
  /// Constructors
  explicit QuantumCircuit(const fs::path &circuit_path = "");
  QuantumCircuit(ModuleOp circuit_module,
                 std::unique_ptr<MLIRContext> context_ptr);
  QuantumCircuit(ModuleOp circuit_module,
                 std::unique_ptr<MLIRContext> context_ptr,
                 unsigned int nr_qubits, unsigned int nr_gates,
                 unsigned int depth);
  /// Copy Constructors
  // Forbid copying the QuantumCircuitEnvironment because the MLIRContext is
  // tied exactly to the circuit module and it is ambiguous if you copy both of
  // them if the copies are then untied from their originals but tied to each
  // other.
  QuantumCircuit(const QuantumCircuit &other) = delete;

  QuantumCircuit &operator=(const QuantumCircuit &other) = delete;

  /// Move Constructors
  QuantumCircuit(QuantumCircuit &&other) noexcept;

  QuantumCircuit &operator=(QuantumCircuit &&) noexcept;

  /// Destructor
  ~QuantumCircuit() = default;

  /// Setters
  bool set(const fs::path &circuit_path = "");
  bool set(ModuleOp circuit_module, std::unique_ptr<MLIRContext> context_ptr);
  bool set(ModuleOp circuit_module, std::unique_ptr<MLIRContext> context_ptr,
           unsigned int nr_qubits, unsigned int nr_gates, unsigned int depth);

  /// Getters
  unsigned int get_nr_qubits() const;
  unsigned int get_nr_gates() const;
  unsigned int get_depth() const;
  std::tuple<unsigned int, unsigned int, unsigned int> get_attributes() const;

  /// Casting
  // NOLINTNEXTLINE to allow implicit conversion to FuncOp.
  operator mlir::func::FuncOp() const;

  /// Short functions
  void clear();
  bool exists() const;
  bool validate();
  bool recompute();

  /**
   *
   * @param pass_index
   * @return
   */
  bool run_pass(unsigned int pass_index);

  /**
   *
   * @param string_stream
   */
  void print(llvm::raw_string_ostream &string_stream) const;
};
} // namespace ai_pass_selector

#endif // QUANTUM_CIRCUIT_HPP