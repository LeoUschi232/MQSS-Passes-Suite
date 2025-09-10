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
#include "mlir_utils.hpp"

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
  switch (circuit_invalid_type(circuit)) {
  case CIRCUIT_VALID:
    break;
  case NO_CIRCUIT:
    std::cerr << "No circuit provided to the environment." << std::endl;
    return;
  case TOO_MANY_QUBITS:
    std::cerr << "Passed circuit has too many qubits." << std::endl;
    return;
  case TOO_MANY_INSTRUCTIONS:
    std::cerr << "Passed circuit has too many instructions." << std::endl;
    return;
  case TOO_LARGE_DEPTH:
    std::cerr << "Passed circuit has too large depth." << std::endl;
    return;
  case NO_QUBIT_ALLOCATIONS:
    std::cerr << "Passed circuit has no qubit allocations." << std::endl;
    return;
  case MULTIPLE_QUBIT_ALLOCATIONS:
    std::cerr << "Passed circuit has multiple qubit allocations." << std::endl;
    return;
  default:
    std::cerr << "Unkown circuit validation error." << std::endl;
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
  if (circuit == nullptr) {
    return {};
  }
  std::unordered_map<std::string, int> circuit_info;
  circuit_info["qubits"] = getNumberOfQubits(func::FuncOp(circuit));
  circuit_info["gates"] = getNumberOfGates(func::FuncOp(circuit));
  circuit_info["depth"] = getCircuitDepth(func::FuncOp(circuit));
  return circuit_info;
}

int QuantumCircuitEnviorment::circuit_invalid_type(ModuleOp circuit) const {
  if (circuit == nullptr) {
    return NO_CIRCUIT;
  }
  std::unordered_map<std::string, int> circuit_info = this->
      get_circuit_info(circuit);
  if (circuit_info["qubits"] > this->max_qubits) {
    return TOO_MANY_QUBITS;
  }
  if (circuit_info["gates"] > this->max_instructions) {
    return TOO_MANY_INSTRUCTIONS;
  }
  if (circuit_info["depth"] > this->max_depth) {
    return TOO_LARGE_DEPTH;
  }
  int nrAllocations = getNumberOfAllocations(func::FuncOp(circuit));
  if (nrAllocations <= 0) {
    return NO_QUBIT_ALLOCATIONS;
  }
  if (nrAllocations >= 2) {
    return MULTIPLE_QUBIT_ALLOCATIONS;
  }
  return CIRCUIT_VALID;
}

std::unordered_map<std::string, int>
QuantumCircuitEnviorment::get_circuit_info() {
  if (this->current_circuit == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return {};
  }
  this->current_circuit.walk([&](Operation *op) {
    std::cout << getOperationName(op) << std::endl;
  });
  return get_circuit_info(this->current_circuit);
}

InstructionBasedTensor<double>
QuantumCircuitEnviorment::get_instruction_based_observation() {
  InstructionBasedTensor<double> observation(
      this->max_qubits, this->max_instructions);
  if (this->current_circuit == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return observation;
  }
  int nrQubits = getNumberOfQubits(func::FuncOp(this->current_circuit));
  if (nrQubits == 0) {
    return observation;
  }

  int instruction_index = 0;
  this->current_circuit.walk([&](Operation *op) {
    if (isMeasurementGate(op)) {
      for (auto operand : op->getOperands()) {
        if (operand.getType().isa<quake::RefType>()) {
          int qubitIndex
              = extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
          if (qubitIndex < 0 || nrQubits <= qubitIndex) {
            continue;
          }
          std::vector<double> no_controls
              = index_to_one_hot(this->max_qubits, -1);
          std::vector<double> target_one_hot
              = index_to_one_hot(this->max_qubits, qubitIndex);
          std::array gate_one_hot
              = GATE_ONE_HOT(getOperationName(op));
          if (isa<quake::MxOp>(op)) {
          }

        }
      }
    } else {
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