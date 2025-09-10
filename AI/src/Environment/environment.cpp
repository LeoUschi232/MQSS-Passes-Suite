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
#include <iostream>

using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
QuantumCircuitEnviorment::QuantumCircuitEnviorment(
    const int max_qubits, const int max_instructions, const int max_depth,
    ModuleOp circuit)
  : max_qubits(max_qubits),
    max_instructions(max_instructions),
    max_depth(max_depth) {
  this->register_quantum_circuit(circuit);
}

void QuantumCircuitEnviorment::register_quantum_circuit(ModuleOp circuit) {
  if (!is_valid_circuit(circuit)) {
    std::cerr << "Invalid circuit passed to env." << std::endl;
    return;
  }
  this->original_circuit = ModuleOp(circuit);
  this->current_circuit = ModuleOp(circuit);
}


std::tuple<InstructionBasedTensor<double>, DepthBasedTensor<double>,
           std::unordered_map<std::string, int> >
QuantumCircuitEnviorment::reset() {
  this->current_circuit = ModuleOp(this->original_circuit);
  return {this->get_instruction_based_observation(),
          this->get_depth_based_observation(), this->get_circuit_info()};
}


std::unordered_map<std::string, int>
QuantumCircuitEnviorment::get_circuit_info(const ModuleOp &circuit) {
  std::unordered_map<std::string, int> circuit_info;
  circuit_info["qubits"] = getNumberOfQubits(func::FuncOp(circuit));
  circuit_info["gates"] = getNumberOfGates(func::FuncOp(circuit));
  circuit_info["depth"] = getCircuitDepth(func::FuncOp(circuit));
  return circuit_info;
}

bool QuantumCircuitEnviorment::is_valid_circuit(ModuleOp circuit) const {
  std::unordered_map<std::string, int> circuit_info
      = this->get_circuit_info(circuit);
  return circuit_info["qubits"] <= this->max_qubits
         && circuit_info["gates"] <= this->max_instructions
         && circuit_info["depth"] <= this->max_depth;
}


std::unordered_map<std::string, int>
QuantumCircuitEnviorment::get_circuit_info() const {
  if (this->current_circuit == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return {};
  }
  return get_circuit_info(this->current_circuit);
}

bool QuantumCircuitEnviorment::is_valid_circuit() const {
  if (this->current_circuit == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return false;
  }
  return this->is_valid_circuit(this->current_circuit);
}

InstructionBasedTensor<double>
QuantumCircuitEnviorment::get_instruction_based_observation() {
  if (this->current_circuit == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return {this->max_qubits, this->max_instructions};
  }
  InstructionBasedTensor<double> observation(
      this->max_qubits, this->max_instructions);

  this->current_circuit.walk([&](Operation *op) {
    if (!isOperatingGate(op)) {
      return;
    }
    if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)) {
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

  });

  return observation;
}

DepthBasedTensor<double>
QuantumCircuitEnviorment::get_depth_based_observation() {
  if (this->current_circuit == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return {this->max_qubits, this->max_depth};
  }
  DepthBasedTensor<double> observation(this->max_qubits, this->max_depth);
  return observation;
}
} // namespace ai_pass_selector