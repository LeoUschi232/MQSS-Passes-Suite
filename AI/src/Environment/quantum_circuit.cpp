#include "Environment/quantum_circuit.hpp"

// Support includes
#include "Support/mlir_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
/// The usages of llvm functions must come before the QuakeOps header which
/// expects them.
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
////////////////////////////////////////////////////////////////////////////////

// Utils includes
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

std::pair<bool, bool>
QuantumCircuit::run_pass(std::unique_ptr<Pass> &pass_ptr) {
  // In cases where the pass is not a AppliedCheckPass, assume by default that
  // the pass was applied if it completes, as we have no way of checking if it
  // was without recomputing the circuit metrics anyway.
  std::shared_ptr<std::atomic_bool> was_applied_ptr = nullptr;
  if (auto *applied_check_pass =
          dynamic_cast<AppliedCheckPass *>(pass_ptr.get())) {
    was_applied_ptr = applied_check_pass->getAppliedPtr();
  }
  try {
    MLIRContext &context = *this->context_ptr.get();
    mlir::PassManager pass_manager(&context);
    pass_manager.addPass(std::move(pass_ptr));
    if (mlir::failed(pass_manager.run(this->circuit_module))) {
      std::cerr << "Pass failed internally." << std::endl;
      this->recompute();
      return {false, false};
    }
  } catch ([[maybe_unused]] const std::runtime_error &error) {
    std::cerr << "Pass threw: " << error_no_stacktrace(error) << std::endl;
    this->recompute();
    return {false, true};
  }

  if (!was_applied_ptr || was_applied_ptr->load()) {
    return {this->recompute(), true};
  }
  // Pass did not apply any changes.
  return {this->validate(), false};
}

std::pair<bool, bool> QuantumCircuit::run_pass(unsigned int pass_index) {
  if (pass_index >= NR_PASSES) {
    return {false, false};
  }
  auto [passname, pass_ptr] = getPassNameAndPointer(pass_index);
  auto [succeeded, was_applied] = this->run_pass(pass_ptr);
  if (!succeeded) {
    std::cerr << "Pass " << passname << " failed." << std::endl;
  }
  return {succeeded, was_applied};
}

QuantumCircuit::operator mlir::func::FuncOp() const {
  return FuncOp(this->circuit_module);
}

void QuantumCircuit::print(llvm::raw_string_ostream &string_stream) const {
  this->circuit_module->print(string_stream);
}

unsigned int QuantumCircuit::getNrQubits() const { return this->nr_qubits; }
unsigned int QuantumCircuit::getNrGates() const { return this->nr_gates; }
unsigned int QuantumCircuit::getDepth() const { return this->depth; }
std::tuple<unsigned int, unsigned int, unsigned int>
QuantumCircuit::get_attributes() const {
  return {this->nr_qubits, this->nr_gates, this->depth};
}

} // namespace ai_pass_selector