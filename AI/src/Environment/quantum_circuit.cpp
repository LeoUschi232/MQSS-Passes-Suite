#include "Environment/quantum_circuit.hpp"

// Utils includes
#include "Support/mlir_utils.hpp"
#include "Utils/info_utils.hpp"
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <iostream>

using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
QuantumCircuit::QuantumCircuit(const fs::path &circuit_path) {
  this->set(circuit_path);
}
QuantumCircuit::QuantumCircuit(ModuleOp circuit_module,
                               std::unique_ptr<MLIRContext> context_ptr) {
  this->set(circuit_module, std::move(context_ptr));
}
QuantumCircuit::QuantumCircuit(ModuleOp circuit_module,
                               std::unique_ptr<MLIRContext> context_ptr,
                               unsigned int nr_qubits, unsigned int nr_gates,
                               unsigned int depth) {
  this->set(circuit_module, std::move(context_ptr), nr_qubits, nr_gates, depth);
}

QuantumCircuit::QuantumCircuit(QuantumCircuit &&other) noexcept
    : circuit_module(std::move(other.circuit_module)),
      context_ptr(std::move(other.context_ptr)), nr_qubits(other.nr_qubits),
      nr_gates(other.nr_gates), depth(other.depth) {}

QuantumCircuit &QuantumCircuit::operator=(QuantumCircuit &&other) noexcept {
  if (this != &other) {
    circuit_module = std::move(other.circuit_module);
    context_ptr = std::move(other.context_ptr);
    nr_qubits = other.nr_qubits;
    nr_gates = other.nr_gates;
    depth = other.depth;
  }
  return *this;
}

void QuantumCircuit::clear() {
  this->context_ptr.reset();
  this->circuit_module = nullptr;
  this->nr_qubits = 0u;
  this->nr_gates = 0u;
  this->depth = 0u;
}

bool QuantumCircuit::set(const fs::path &circuit_path) {
  this->clear();
  if (circuit_path.empty()) {
    // Assume empty construction for later assignment.
    return false;
  }
  const std::string circuit_text = readFileToString(circuit_path.string());
  if (circuit_text.empty()) {
    std::cerr << "Failed to read circuit file: " << circuit_path << std::endl;
    return false;
  }
  auto [circuit, context] = extractModuleOpAndContextPointer(circuit_text);
  if (!context || !*context) {
    return false;
  }
  MLIRContext *raw_context = *std::move(context);
  context.reset();
  context_ptr.reset(raw_context);
  circuit_module = circuit;
  return recompute();
}
bool QuantumCircuit::set(ModuleOp circuit_module,
                         std::unique_ptr<MLIRContext> context_ptr) {
  this->context_ptr = std::move(context_ptr);
  this->circuit_module = circuit_module;
  return this->recompute();
}

bool QuantumCircuit::set(ModuleOp circuit_module,
                         std::unique_ptr<MLIRContext> context_ptr,
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

bool QuantumCircuit::exists() const {
  return this->circuit_module != nullptr && this->context_ptr != nullptr &&
         this->nr_qubits >= GLOBAL_MIN_NR_QUBITS &&
         this->nr_gates >= GLOBAL_MIN_NR_GATES && this->depth > 0u;
}
bool QuantumCircuit::validate() {
  // Validate is sort-of like exists but clears if invalid.
  if (!this->exists()) {
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
    MLIRContext &context = *this->context_ptr.get();
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
QuantumCircuit::operator mlir::func::FuncOp() const {
  return FuncOp(this->circuit_module);
}

void QuantumCircuit::print(llvm::raw_string_ostream &string_stream) const {
  this->circuit_module->print(string_stream);
}

unsigned int QuantumCircuit::get_nr_qubits() const { return this->nr_qubits; }
unsigned int QuantumCircuit::get_nr_gates() const { return this->nr_gates; }
unsigned int QuantumCircuit::get_depth() const { return this->depth; }
std::tuple<unsigned int, unsigned int, unsigned int>
QuantumCircuit::get_attributes() const {
  return {this->nr_qubits, this->nr_gates, this->depth};
}

} // namespace ai_pass_selector