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
  case AMBIGUOUS_MEASUREMENT:
    std::cerr << "Passed circuit has ambiguous measurements." << std::endl;
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
  bool ambiguous_measurement = false;
  circuit.walk([&](Operation *op) {
    if (isMeasurementGate(op) && op->getOpOperands().size() != 1) {
      ambiguous_measurement = true;
    }
  });
  if (ambiguous_measurement) {
    return AMBIGUOUS_MEASUREMENT;
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
  const int NR_QUBITS = getNumberOfQubits(FuncOp(this->current_circuit));
  if (NR_QUBITS == 0) {
    return observation;
  }

  const int MAX_QUBITS = this->max_qubits;
  const int GATE_OFFSET = MAX_QUBITS;
  const int PARAM_OFFSET = GATE_OFFSET + NR_GATES;

  int instruction_index = 0;
  this->current_circuit.walk([&](Operation *op) {
    if (!isOperatingGate(op)) {
      return;
    }
    if (instruction_index >= this->max_instructions) {
      throw std::runtime_error("instruction_index exceeded max_instructions.");
    }
    std::string gate_name = getOnlyGateName(op);
    int gate_index = GATE_INDEX(gate_name);
    if (gate_index < 0) {
      throw std::runtime_error("Gate: " + gate_name);
    }

    std::vector<int> controls = {};
    std::vector<int> targets = {};
    std::vector params(MAX_GATE_PARAMS, 0.0);
    bool isAdj = false;

    if (isMeasurementGate(op)) {
      targets = getMeasurementTargets(op, NR_QUBITS);
    } else {
      std::tie(controls, targets, params, isAdj)
          = getOperatingControlsTargetsParams(op);
    }

    double *row = observation.row_ptr(instruction_index);
    std::fill_n(row, observation.shape[1], 0.0);

    // Controls
    for (int qubit : controls) {
      if (0 <= qubit && qubit < MAX_QUBITS) {
        row[qubit] = -1.0;
      }
    }

    // Targets
    for (int qubit : targets) {
      if (0 <= qubit && qubit < MAX_QUBITS) {
        row[qubit] = 1.0;
      }
    }

    // Gate
    row[GATE_OFFSET + gate_index] = isAdj ? -1.0 : 1.0;

    // params: [angle1, angle2, angle3]
    for (int i = 0; i < MAX_GATE_PARAMS; i++) {
      row[PARAM_OFFSET + i] = params[i];
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

  const int NR_QUBITS = getNumberOfQubits(FuncOp(this->current_circuit));
  if (NR_QUBITS == 0) {
    return observation;
  }

  constexpr int GATE_OFFSET = 0;
  constexpr int PARAM_OFFSET = GATE_OFFSET + NR_GATES;
  constexpr int CONTROL_INFO_OFFSET = PARAM_OFFSET + MAX_GATE_PARAMS;
  constexpr int EXTRAS_OFFSET = CONTROL_INFO_OFFSET + QUBIT_ROLE;

  // Greedy ASAP schedule: track next free depth per qubit.
  std::vector next_free_depth(NR_QUBITS, 0);

  this->current_circuit.walk([&](Operation *op) {
    if (!isOperatingGate(op)) {
      return;
    }

    std::string gate_name = getOnlyGateName(op);
    int gate_index = GATE_INDEX(gate_name);
    if (gate_index < 0) {
      throw std::runtime_error("Gate: " + gate_name);
    }
    std::vector<int> controls = {};
    std::vector<int> targets = {};
    std::vector params(MAX_GATE_PARAMS, 0.0);
    bool isAdj = false;

    if (isMeasurementGate(op)) {
      targets = getMeasurementTargets(op, NR_QUBITS);
    } else {
      std::tie(controls, targets, params, isAdj)
          = getOperatingControlsTargetsParams(op);
    }

    // Determine layer = depth cross-section
    int scheduled_depth = 0;
    for (int qubit : controls) {
      scheduled_depth = std::max(scheduled_depth, next_free_depth[qubit]);
    }
    for (int qubit : targets) {
      scheduled_depth = std::max(scheduled_depth, next_free_depth[qubit]);
    }

    if (scheduled_depth >= this->max_depth) {
      throw std::runtime_error(
          "Depth Based Observation exceeds configured max_depth.");
    }

    // Populate features for every involved qubit at this depth
    auto write_cell = [&](int qubit_index, bool is_control_qubit) {
      double *cell = observation.cell_ptr(scheduled_depth, qubit_index);

      // The tensor storage is value-initialized to 0.0; write only non-zeros.
      if (gate_index >= 0) {
        cell[GATE_OFFSET + gate_index] = isAdj ? -1.0 : 1.0;
      }

      // params: [adjoint, angle1, angle2, angle3]
      for (int i = 0; i < static_cast<int>(params.size()); i++) {
        cell[PARAM_OFFSET + i] = params[i];
      }

      // control info: [is_control_qubit, is_target_qubit]
      cell[CONTROL_INFO_OFFSET] = is_control_qubit ? -1.0 : 1.0;
      for (int t : targets) {
        cell[EXTRAS_OFFSET + t] = 1.0;
      }
      for (int c : controls) {
        cell[EXTRAS_OFFSET + c] = -1.0;
      }
    };
    std::vector<char> touched(NR_QUBITS, 0);
    for (int qubit : targets) {
      if (!touched[qubit]) {
        write_cell(qubit, false);
        touched[qubit] = 1;
      }
    }
    for (int qubit : controls) {
      if (!touched[qubit]) {
        write_cell(qubit, true);
        touched[qubit] = 1;
      }
    }

    // Advance next free depth for all qubits touched by this op
    int new_depth = scheduled_depth + 1;
    for (int qubit : targets) {
      next_free_depth[qubit] = std::max(next_free_depth[qubit], new_depth);
    }
    for (int qubit : controls) {
      next_free_depth[qubit] = std::max(next_free_depth[qubit], new_depth);
    }
  });

  return observation;
}


std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, bool>
QuantumCircuitEnviorment::getOperatingControlsTargetsParams(Operation *op) {
  if (isMeasurementGate(op) || !isOperatingGate(op)) {
    return {{}, {}, {}, false};
  }
  std::vector params(MAX_GATE_PARAMS, 0.0);
  std::string gate_name = getOnlyGateName(op);
  auto gateOp = dyn_cast<quake::OperatorInterface>(op);
  if (!gateOp) {
    throw std::runtime_error("Non-operator-interface for: " + gate_name);
  }
  std::vector<int> targets = getIndicesOfValueRange(gateOp.getTargets());
  std::vector<int> controls = getIndicesOfValueRange(gateOp.getControls());
  auto param_values = getParametersValues(gateOp.getParameters());
  auto angles = params_to_angles(param_values);
  if (angles.size() > MAX_GATE_PARAMS) {
    throw std::runtime_error("Detected gate with too many angles.");
  }
  for (std::size_t i = 0; i < angles.size(); i++) {
    params[i] = angles[i];
  }
  return {controls, targets, params, gateOp.isAdj()};
}


} // namespace ai_pass_selector