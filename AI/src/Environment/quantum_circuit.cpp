#include "Environment/quantum_circuit.hpp"

// Utils includes
#include "Support/mlir_utils.hpp"
#include "Utils/info_utils.hpp"
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <iostream>

using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
void QuantumCircuit::clear() {
  if (context_ptr && this->context_ptr.get() != nullptr) {
    delete *this->context_ptr.get();
  }
  this->context_ptr = nullptr;
  nr_qubits = 0u;
  nr_gates = 0u;
  depth = 0u;
}

bool QuantumCircuit::set(const fs::path &circuit_path) {
  this->clear();
  if (circuit_path.empty()) {
    // Assume construction of environment for later circuit registration.
    return false;
  }
  const std::string circuit_text = readFileToString(circuit_path.string());
  if (circuit_text.empty()) {
    std::cerr << "Failed to read circuit file: " << circuit_path << std::endl;
    return false;
  }
  auto [circuit, context] = extractModuleOpAndContextPointer(circuit_text);
  return this->set(circuit, std::move(context));
}
bool QuantumCircuit::set(ModuleOp circuit_module,
                         std::unique_ptr<MLIRContext *> context_ptr) {
  this->context_ptr = std::move(context_ptr);
  this->circuit_module = circuit_module;
  return this->recompute();
}

bool QuantumCircuit::set(ModuleOp circuit_module,
                         std::unique_ptr<MLIRContext *> context_ptr,
                         unsigned int nr_qubits, unsigned int nr_gates,
                         unsigned int depth) {
  // Assume that if the values are provided they are correct against the
  // provided circuit module.
  // It is possible to provide garbage values for nr_qubits, nr_gates and depth
  // that are not true for the circuit module which would cause UB, but this
  // method exists specifically so that expensive computation of nr_qubits,
  // nr_gates and depth does not need to be performed again if the caller knows
  // the true values.
  this->context_ptr = std::move(context_ptr);
  this->circuit_module = circuit_module;
  this->nr_qubits = nr_qubits;
  this->nr_gates = nr_gates;
  this->depth = depth;
  return this->validate();
}

bool QuantumCircuit::validate() {
  if (this->nr_qubits < GLOBAL_MIN_NR_QUBITS ||
      this->nr_gates < GLOBAL_MIN_NR_GATES || this->depth <= 0u ||
      this->circuit_module == nullptr || this->context_ptr == nullptr) {
    this->clear();
    return false;
  }
  return true;
}

bool QuantumCircuit::recompute() {
  if (this->circuit_module == nullptr || this->context_ptr == nullptr) {
    this->clear();
    return false;
  }
  std::tie(nr_qubits, nr_gates, depth) =
      getQubitsInstructionsDepth(FuncOp(this->circuit_module));
  return this->validate();
}

bool QuantumCircuit::run_pass(unsigned int pass_index) {
  if (pass_index >= NR_PASSES) {
    return false;
  }
  auto [passname, passptr] = getPassNameAndPointer(pass_index);
  try {
    // Variable context must be a MLIRContext&.
    // Types are:
    // context_ptr = unique_ptr<MLIRContext*>
    // context_ptr.get() = MLIRContext**
    // *context_ptr.get() = MLIRContext*
    // **context_ptr.get() = MLIRContext
    MLIRContext &context = **this->context_ptr.get();
    mlir::PassManager pass_manager(&context);
    pass_manager.addPass(std::move(passptr));
    if (mlir::failed(pass_manager.run(this->circuit_module))) {
      this->recompute();
      return false;
    }
  } catch (const std::runtime_error &error) {
    std::cerr << "Pass " << passname << " failed with " << error.what()
              << std::endl;
    this->recompute();
    return false;
  }
  return this->recompute();
}

} // namespace ai_pass_selector