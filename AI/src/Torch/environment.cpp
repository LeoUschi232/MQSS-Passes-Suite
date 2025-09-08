#include "Torch/environment.hpp"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// Passes includes
#include "Passes/Cancellations.hpp"

// Support includes
#include "Support/CodeGen/Quake.hpp"

// Torch includes
#include <torch/torch.h>

// Standard library includes
#include <string>
#include <unordered_map>
#include <utility>

using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
QuantumCircuitEnviorment::QuantumCircuitEnviorment(
    const int max_qubits, const int max_instructions, const int max_depth,
    ModuleOp circuit)
  : max_qubits(max_qubits),
    max_instructions(max_instructions),
    max_depth(max_depth),
    original_circuit(ModuleOp(circuit)),
    current_circuit(ModuleOp(circuit)) {
}


std::pair<torch::Tensor, std::unordered_map<std::string, int> >
QuantumCircuitEnviorment::reset(int seed) {
  this->current_circuit = ModuleOp(this->original_circuit);
  return {torch::rand({1, this->max_qubits * this->max_instructions}),
          this->get_circuit_info()};
}

int QuantumCircuitEnviorment::printOperation(Operation *op) {
  if (op->getDialect()->getNamespace() != "quake") {
    return 0;
  }
  std::cout << std::string(op->getName().getStringRef()) << std::endl;
  return 1;
}

std::unordered_map<std::string, int>
QuantumCircuitEnviorment::get_circuit_info() {
  std::unordered_map<std::string, int> circuit_info;
  this->current_circuit.walk([&](quake::AllocaOp allocOp) {
    if (allocOp.getType().dyn_cast<quake::RefType>()) {
      std::cout << "+1" << std::endl;
    } else if (auto qvecType = allocOp.getType().dyn_cast<quake::VeqType>()) {
      std::cout << qvecType.getSize() << std::endl;
    }
  });
  this->printOperation(this->current_circuit);
  circuit_info["qubits"] = 0;
  circuit_info["gates"] = 0;
  circuit_info["depth"] = 0;
  return circuit_info;
}


} // namespace ai_pass_selector