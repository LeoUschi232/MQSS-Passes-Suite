#include "Environment/environment.hpp"

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"

// Passes includes
#include "Passes/Cancellations.hpp"

// Support includes
#include "Support/CodeGen/Quake.hpp"

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


std::pair<QuantumCircuitTensor<double>, std::unordered_map<std::string, int> >
QuantumCircuitEnviorment::reset(int seed) {
  this->current_circuit = ModuleOp(this->original_circuit);
  return {{seed, seed, seed}, this->get_circuit_info()};
}


std::unordered_map<std::string, int>
QuantumCircuitEnviorment::get_circuit_info(ModuleOp circuit) {
  std::unordered_map<std::string, int> circuit_info;
  circuit_info["qubits"] = getNumberOfQubits(func::FuncOp(circuit));
  circuit_info["gates"] = getNumberOfGates(func::FuncOp(circuit));
  circuit_info["depth"] = getCircuitDepth(func::FuncOp(circuit));
  return circuit_info;
}


std::unordered_map<std::string, int>
QuantumCircuitEnviorment::get_circuit_info() const {
  return get_circuit_info(this->current_circuit);
}

bool QuantumCircuitEnviorment::is_valid_circuit() const {
  std::unordered_map<std::string, int> circuit_info = this->get_circuit_info();
  return circuit_info["qubits"] <= this->max_qubits
         && circuit_info["gates"] <= this->max_instructions
         && circuit_info["depth"] <= this->max_depth;
}

QuantumCircuitTensor<double> QuantumCircuitEnviorment::get_observation() {
  int mock_nr_supported_gates = 10;
  QuantumCircuitTensor<double> observation(
      this->max_qubits, this->max_depth, mock_nr_supported_gates);


  std::vector depths(nrQubits, 0);
  circuit.walk([&](Operation *op) {
    if (op->getDialect()->getNamespace() == "quake"
        && !isa<quake::AllocaOp>(op)
        && !isa<quake::ExtractRefOp>(op)) {
      if (isa<quake::MxOp>(op)
          || isa<quake::MyOp>(op)
          || isa<quake::MzOp>(op)) {
        for (auto operand : op->getOperands()) {
          // Check if it's qubit reference
          if (operand.getType().isa<quake::RefType>()) {
            int qubitIndex =
                extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
            if (0 <= qubitIndex && qubitIndex < nrQubits) {
              depths[qubitIndex]++;
            }
          } else if (operand.getType().isa<quake::VeqType>()) {
            for (int qubitIndex = 0; qubitIndex < nrQubits; qubitIndex++) {
              depths[qubitIndex]++;
            }
          }
        }
      } else {
        auto gate = dyn_cast<quake::OperatorInterface>(op);
        std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
        std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
        targets.insert(targets.end(), controls.begin(), controls.end());
        int max_depth = 0;
        for (int qubit : targets) {
          max_depth = std::max(max_depth, depths[qubit]);
        }
        for (int qubit : targets) {
          depths[qubit] = max_depth + 1;
        }
      }
    }
  });
  return *std::ranges::max_element(depths);
}
} // namespace ai_pass_selector