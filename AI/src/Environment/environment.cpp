#include "Environment/environment.hpp"

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// MLIR includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

// Cudaq includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

// Utils includes
#include "Support/mlir_utils.hpp"
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::ModuleOp;
using mlir::Operation;
using mlir::func::FuncOp;
////////////////////////////////////////////////////////////////////////////////

using namespace mqss::support::quakeDialect;
namespace fs = std::filesystem;

namespace ai_pass_selector {

QuantumCircuitEnviorment::QuantumCircuitEnviorment(
    const unsigned int max_qubits, const fs::path &circuit_path,
    unsigned int max_steps)
    : max_qubits(max_qubits), circuit_path(circuit_path), max_steps(max_steps),
      current_step(0) {
  if (!circuit_path.empty()) {
    this->register_quantum_circuit(circuit_path);
  }
}

void QuantumCircuitEnviorment::clear_circuit() {
  this->circuit_path.clear();
  this->circuit_module = nullptr;
  this->context_ptr = nullptr;
}

bool QuantumCircuitEnviorment::register_quantum_circuit(
    const fs::path &circuit_path) {
  if (circuit_path.empty()) {
    // Assume construction of environment for later circuit registration.
    return false;
  }
  const std::string circuit_text = readFileToString(circuit_path.string());
  if (circuit_text.empty()) {
    std::cerr << "Failed to read circuit file: " << circuit_path << std::endl;
    return false;
  }
  this->circuit_path = circuit_path;

  auto [circuit, context] = extractModuleOpAndContextPointer(circuit_text);
  // Hold the context pointer so there is no segfault when accessing circuit.
  this->context_ptr = std::move(context);

  switch (circuit_invalid_type(FuncOp(circuit))) {
  case CIRCUIT_VALID:
    break;
  case NO_CIRCUIT:
    std::cerr << "No circuit provided to the environment." << std::endl;
    return false;
  case TOO_MANY_QUBITS:
    std::cerr << "Passed circuit has too many qubits." << std::endl;
    return false;
  case NO_QUBIT_ALLOCATIONS:
    std::cerr << "Passed circuit has no qubit allocations." << std::endl;
    return false;
  case MULTIPLE_QUBIT_ALLOCATIONS:
    std::cerr << "Passed circuit has multiple qubit allocations." << std::endl;
    return false;
  case AMBIGUOUS_MEASUREMENT:
    std::cerr << "Passed circuit has ambiguous measurements." << std::endl;
    return false;
  default:
    std::cerr << "Unkown circuit validation error." << std::endl;
    return false;
  }

  this->circuit_module = circuit;
  this->current_step = 0;
  return true;
}

void QuantumCircuitEnviorment::reset() {
  this->register_quantum_circuit(this->circuit_path);
}

unsigned int
QuantumCircuitEnviorment::circuit_invalid_type(FuncOp circuit) const {
  if (circuit == nullptr) {
    return NO_CIRCUIT;
  }
  unsigned int nr_qubits = 0;
  unsigned int nr_allocations = 0;
  bool ambiguous_measurement = false;
  circuit.walk([&](Operation *op) -> mlir::WalkResult {
    if (nr_allocations >= 2 || ambiguous_measurement ||
        nr_qubits > max_qubits) {
      return mlir::WalkResult::interrupt();
    }
    if (isa<quake::AllocaOp>(op)) {
      nr_allocations++;
      if (nr_allocations >= 2) {
        return mlir::WalkResult::interrupt();
      }
      if (auto allocOp = dyn_cast<quake::AllocaOp>(op);
          allocOp.getType().dyn_cast<quake::RefType>()) {
        nr_qubits += 1;
      } else if (auto qvecType = allocOp.getType().dyn_cast<quake::VeqType>()) {
        nr_qubits += qvecType.getSize();
      }
      if (nr_qubits > max_qubits) {
        return mlir::WalkResult::interrupt();
      }
    } else if (isMeasurementGate(op) && op->getOpOperands().size() != 1) {
      ambiguous_measurement = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (nr_qubits > max_qubits) {
    return TOO_MANY_QUBITS;
  }
  if (nr_allocations <= 0) {
    return NO_QUBIT_ALLOCATIONS;
  }
  if (nr_allocations >= 2) {
    return MULTIPLE_QUBIT_ALLOCATIONS;
  }
  if (ambiguous_measurement) {
    return AMBIGUOUS_MEASUREMENT;
  }
  return CIRCUIT_VALID;
}

std::unordered_map<std::string, unsigned int>
QuantumCircuitEnviorment::get_circuit_info(FuncOp circuit) {
  if (circuit == nullptr) {
    return {};
  }
  auto [nrQubits, nrGates, depth] = getQubitsInstructionsDepth(circuit);
  return {{"qubits", nrQubits}, {"gates", nrGates}, {"depth", depth}};
}

std::unordered_map<std::string, unsigned int>
QuantumCircuitEnviorment::get_circuit_info() const {
  if (this->circuit_module == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return {};
  }
  return get_circuit_info(FuncOp(this->circuit_module));
}

std::tuple<double, bool> QuantumCircuitEnviorment::step(unsigned int action) {
  if (this->circuit_module == nullptr) {
    throw std::runtime_error("No circuit registered in the environment.");
  }
  if (action >= NR_PASSES) {
    throw std::runtime_error("Invalid action: " + std::to_string(action));
  }
  std::unordered_map<std::string, unsigned int> previous_circuit_info =
      this->get_circuit_info();
  double previous_depth = previous_circuit_info["depth"];
  double previous_gates = previous_circuit_info["gates"];

  std::unique_ptr<mlir::Pass> pass = PASS_FUNCTIONS[action]();

  MLIRContext &context = **this->context_ptr.get();
  mlir::PassManager pass_manager(&context);
  pass_manager.addPass(std::move(pass));
  pass_manager.addPass(mlir::createCanonicalizerPass());
  pass_manager.addPass(mlir::createCSEPass());

  if (mlir::failed(pass_manager.run(this->circuit_module))) {
    throw std::runtime_error("Pass manager failed.");
  }

  std::unordered_map<std::string, unsigned int> current_circuit_info =
      this->get_circuit_info();

  double current_depth = current_circuit_info["depth"];
  double current_gates = current_circuit_info["gates"];
  return {previous_depth - current_depth + previous_gates - current_gates,
          ++this->current_step >= this->max_steps};
}

InstructionsTensor<double>
QuantumCircuitEnviorment::get_instruction_based_observation() {
  InstructionsTensor<double> observation(this->max_qubits);
  if (this->circuit_module == nullptr) {
    return observation;
  }
  const int NR_QUBITS = getNumberOfQubits(FuncOp(this->circuit_module));
  if (NR_QUBITS == 0) {
    return observation;
  }

  const int MAX_QUBITS = this->max_qubits;
  const int GATE_OFFSET = MAX_QUBITS;
  const int PARAM_OFFSET = GATE_OFFSET + NR_GATES;

  int instruction_index = 0;
  this->circuit_module.walk([&](Operation *op) {
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
      std::tie(controls, targets, params, isAdj) =
          getOperatingControlsTargetsParams(op);
    }

    // TODO: Fix this
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

DepthsTensor<double> QuantumCircuitEnviorment::get_depth_based_observation() {
  DepthsTensor<double> observation(this->max_qubits);
  if (this->circuit_module == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return observation;
  }

  const int NR_QUBITS = getNumberOfQubits(FuncOp(this->circuit_module));
  if (NR_QUBITS == 0) {
    return observation;
  }

  constexpr int GATE_OFFSET = 0;
  constexpr int PARAM_OFFSET = GATE_OFFSET + NR_GATES;
  constexpr int CONTROL_INFO_OFFSET = PARAM_OFFSET + MAX_GATE_PARAMS;
  constexpr int EXTRAS_OFFSET = CONTROL_INFO_OFFSET + QUBIT_ROLE;

  // Greedy ASAP schedule: track next free depth per qubit.
  std::vector next_free_depth(NR_QUBITS, 0);

  this->circuit_module.walk([&](Operation *op) {
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
      std::tie(controls, targets, params, isAdj) =
          getOperatingControlsTargetsParams(op);
    }

    // Determine layer = depth cross-section
    int scheduled_depth = 0;
    for (int qubit : controls) {
      scheduled_depth = std::max(scheduled_depth, next_free_depth[qubit]);
    }
    for (int qubit : targets) {
      scheduled_depth = std::max(scheduled_depth, next_free_depth[qubit]);
    }

    // Populate features for every involved qubit at this depth
    auto write_cell = [&](int qubit_index, bool is_control_qubit) {
      // TODO: Fix this
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