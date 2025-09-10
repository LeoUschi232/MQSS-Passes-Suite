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


////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::ModuleOp;
using mlir::func::FuncOp;
////////////////////////////////////////////////////////////////////////////////

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
  circuit_info["qubits"] = getNumberOfQubits(FuncOp(circuit));
  circuit_info["gates"] = getNumberOfGates(FuncOp(circuit));
  circuit_info["depth"] = getCircuitDepth(FuncOp(circuit));
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
  int nrAllocations = getNumberOfAllocations(FuncOp(circuit));
  if (nrAllocations <= 0) {
    return NO_QUBIT_ALLOCATIONS;
  }
  if (nrAllocations >= 2) {
    return MULTIPLE_QUBIT_ALLOCATIONS;
  }
  return CIRCUIT_VALID;
}

std::unordered_map<std::string, int>
QuantumCircuitEnviorment::get_circuit_info() const {
  if (this->current_circuit == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return {};
  }
  return get_circuit_info(this->current_circuit);
}

InstructionBasedTensor<double>
QuantumCircuitEnviorment::get_instruction_based_observation() {
  InstructionBasedTensor<double> observation(
      this->max_qubits, this->max_instructions);
  if (this->current_circuit == nullptr) {
    return observation;
  }
  if (getNumberOfQubits(FuncOp(this->current_circuit)) == 0) {
    return observation;
  }

  const int MAX_QUBITS = this->max_qubits;
  const int GATE_OFFSET = 2 * MAX_QUBITS;
  const int PARAM_OFFSET = GATE_OFFSET + NR_GATES;

  int instruction_index = 0;
  this->current_circuit.walk([&](Operation *op) {
    if (!isOperatingGate(op)) {
      return;
    }
    auto gateOp = dyn_cast<quake::OperatorInterface>(op);
    double *row = observation.row_ptr(instruction_index);
    std::fill_n(row, observation.shape[1], 0.0);

    // Controls
    for (int qubit : getIndicesOfValueRange(gateOp.getControls())) {
      if (0 <= qubit && qubit < MAX_QUBITS) {
        row[qubit] = 1.0;
      }
    }

    // Targets
    for (int qubit : getIndicesOfValueRange(gateOp.getTargets())) {
      if (0 <= qubit && qubit < MAX_QUBITS) {
        row[MAX_QUBITS + qubit] = 1.0;
      }
    }

    // Gate
    if (int gate_index = GATE_INDEX(getOnlyGateName(op)); gate_index >= 0) {
      row[GATE_OFFSET + gate_index] = 1.0;
    }

    // params: [adjoint, angle1, angle2, angle3]
    row[PARAM_OFFSET] = gateOp.isAdj() ? 1.0 : 0.0;
    auto param_values = getParametersValues(gateOp.getParameters());
    if (param_values.size() > MAX_GATE_ANGLES) {
      throw std::runtime_error("Detected gate with too many angles.");
    }
    auto angles = params_to_angles(param_values);
    for (std::size_t i = 0; i < angles.size() && i < MAX_GATE_ANGLES; ++i) {
      row[PARAM_OFFSET + 1 + i] = angles[i];
    }
    instruction_index++;
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

  int number_of_qubits = getNumberOfQubits(FuncOp(this->current_circuit));
  if (number_of_qubits == 0) {
    return observation;
  }

  constexpr int feature_gate_offset = 0;
  constexpr int feature_param_offset = feature_gate_offset + NR_GATES;
  constexpr int feature_control_info_offset =
      feature_param_offset + MAX_GATE_PARAMS;
  constexpr int feature_targets_offset =
      feature_control_info_offset + CONTROL_PARAMS;

  // Greedy ASAP schedule: track next free depth per qubit.
  std::vector next_free_depth(number_of_qubits, 0);

  this->current_circuit.walk([&](Operation *op) {
    if (!isOperatingGate(op)) {
      return;
    }
    auto gate_op = dyn_cast<quake::OperatorInterface>(op);

    std::vector<int> control_indices = getIndicesOfValueRange(
        gate_op.getControls());
    std::vector<int> target_indices = getIndicesOfValueRange(
        gate_op.getTargets());

    // Filter invalid indices just in case
    control_indices.erase(
        std::remove_if(
            control_indices.begin(),
            control_indices.end(),
            [&](int q) { return q < 0 || q >= number_of_qubits; }),
        control_indices.end());
    target_indices.erase(
        std::remove_if(
            target_indices.begin(),
            target_indices.end(),
            [&](int q) { return q < 0 || q >= number_of_qubits; }),
        target_indices.end());

    // Determine layer = depth cross-section
    int scheduled_depth = 0;
    for (int qubit : control_indices) {
      scheduled_depth = std::max(scheduled_depth, next_free_depth[qubit]);
    }
    for (int qubit : target_indices) {
      scheduled_depth = std::max(scheduled_depth, next_free_depth[qubit]);
    }

    if (scheduled_depth >= this->max_depth) {
      throw std::runtime_error(
          "Depth Based Observation exceeds configured max_depth.");
    }

    // Common gate features
    int gate_index = GATE_INDEX(getOnlyGateName(op));
    auto param_values = getParametersValues(gate_op.getParameters());
    if (param_values.size() > MAX_GATE_ANGLES) {
      throw std::runtime_error("Detected gate with too many angles.");
    }
    auto angles = params_to_angles(param_values);

    // Populate features for every involved qubit at this depth
    auto write_cell = [&](int qubit_index, bool is_control_qubit) {
      double *cell = observation.cell_ptr(scheduled_depth, qubit_index);

      // The tensor storage is value-initialized to 0.0; write only non-zeros.
      if (gate_index >= 0) {
        cell[feature_gate_offset + gate_index] = 1.0;
      }

      // params: [adjoint, angle1, angle2, angle3]
      cell[feature_param_offset] = gate_op.isAdj() ? 1.0 : 0.0;
      for (int i = 0; i < static_cast<int>(angles.size()); i++) {
        cell[feature_param_offset + 1 + i] = angles[i];
      }

      // control info: [is_control_qubit, is_target_qubit]
      cell[feature_control_info_offset] = is_control_qubit ? 1.0 : 0.0;
      cell[feature_control_info_offset + 1] = is_control_qubit ? 0.0 : 1.0;

      // targets (only if gate is controlled)
      if (is_control_qubit) {
        for (int t : target_indices) {
          cell[feature_targets_offset + t] = 1.0;
        }
      } else {
        for (int c : control_indices) {
          cell[feature_targets_offset + c] = 1.0;
        }
      }
    };
    std::vector<char> touched(number_of_qubits, 0);
    for (int qubit : target_indices) {
      if (!touched[qubit]) {
        write_cell(qubit, false);
        touched[qubit] = 1;
      }
    }
    for (int qubit : control_indices) {
      if (!touched[qubit]) {
        write_cell(qubit, true);
        touched[qubit] = 1;
      }
    }

    // Advance next free depth for all qubits touched by this op
    int new_depth = scheduled_depth + 1;
    for (int q : target_indices) {
      next_free_depth[q] = std::max(next_free_depth[q], new_depth);
    }
    for (int q : control_indices) {
      next_free_depth[q] = std::max(next_free_depth[q], new_depth);
    }
  });

  return observation;
}

} // namespace ai_pass_selector