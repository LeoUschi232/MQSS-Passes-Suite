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
#include <Environment/random_quantum_circuit_generator.hpp>
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

QuantumCircuitEnvironment::QuantumCircuitEnvironment(
    unsigned int max_qubits, unsigned int max_steps,
    const fs::path &circuit_path)
    : max_qubits(max_qubits), circuit_path(circuit_path), context_ptr(nullptr),
      max_steps(max_steps), current_step(0) {
  if (!circuit_path.empty()) {
    this->register_quantum_circuit(circuit_path);
  }
}

QuantumCircuitEnvironment::QuantumCircuitEnvironment(
    QuantumCircuitEnvironment &&other) noexcept
    : max_qubits(other.max_qubits), circuit_path(std::move(other.circuit_path)),
      circuit_module(std::move(other.circuit_module)),
      context_ptr(std::move(other.context_ptr)), max_steps(other.max_steps),
      current_step(other.current_step),
      qubits_and_gates_distribution_params(
          std::move(other.qubits_and_gates_distribution_params)),
      gates_weights(std::move(other.gates_weights)) {}

QuantumCircuitEnvironment &QuantumCircuitEnvironment::operator=(
    QuantumCircuitEnvironment &&other) noexcept {
  if (this != &other) {
    if (context_ptr && this->context_ptr.get() != nullptr) {
      delete *this->context_ptr.get();
    }
    max_qubits = other.max_qubits;
    circuit_path = std::move(other.circuit_path);
    circuit_module = std::move(other.circuit_module);
    context_ptr = std::move(other.context_ptr);
    max_steps = other.max_steps;
    current_step = other.current_step;
    qubits_and_gates_distribution_params =
        std::move(other.qubits_and_gates_distribution_params);
    gates_weights = std::move(other.gates_weights);
  }
  return *this;
}

void QuantumCircuitEnvironment::clear() {
  this->circuit_path.clear();
  this->circuit_module = nullptr;
  if (context_ptr && this->context_ptr.get() != nullptr) {
    delete *this->context_ptr.get();
  }
  this->context_ptr = nullptr;
  this->qubits_and_gates_distribution_params = std::nullopt;
  this->gates_weights = std::nullopt;
  this->current_step = 0;
}

void QuantumCircuitEnvironment::reset() {
  if (!this->circuit_path.empty()) {
    this->register_quantum_circuit(this->circuit_path);
    return;
  }
  if (qubits_and_gates_distribution_params.has_value() &&
      gates_weights.has_value()) {
    // Cap nr of qubits but don't cap instructions.
    auto [module, context] = random_quantum_circuit_from_embedded_statistics(
        qubits_and_gates_distribution_params.value(), gates_weights.value(),
        {2, max_qubits});
    this->clear();
    this->circuit_module = module;
    this->context_ptr =
        std::make_unique<MLIRContext *>(std::move(context).release());
    return;
  }
  std::cerr << "No circuit or randomization parameters provided to reset."
            << std::endl;
}

bool QuantumCircuitEnvironment::register_quantum_circuit(
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

  // Hold the context pointer so there is no segfault when accessing circuit.
  auto [circuit, context] = extractModuleOpAndContextPointer(circuit_text);
  this->clear();
  this->circuit_path = circuit_path;
  this->context_ptr = std::move(context);

  switch (circuit_invalid_type(FuncOp(circuit))) {
  case CIRCUIT_VALID:
    break;
  case NO_CIRCUIT:
    this->clear();
    std::cerr << "No circuit provided to the environment." << std::endl;
    return false;
  case TOO_MANY_QUBITS:
    this->clear();
    std::cerr << "Passed circuit has too many qubits." << std::endl;
    return false;
  case NO_QUBIT_ALLOCATIONS:
    this->clear();
    std::cerr << "Passed circuit has no qubit allocations." << std::endl;
    return false;
  case MULTIPLE_QUBIT_ALLOCATIONS:
    this->clear();
    std::cerr << "Passed circuit has multiple qubit allocations." << std::endl;
    return false;
  case AMBIGUOUS_MEASUREMENT:
    this->clear();
    std::cerr << "Passed circuit has ambiguous measurements." << std::endl;
    return false;
  default:
    this->clear();
    std::cerr << "Unkown circuit validation error." << std::endl;
    return false;
  }
  this->circuit_module = circuit;
  return true;
}

void QuantumCircuitEnvironment::register_randomizer_params(
    const std::tuple<double, double, double, double, double>
        &qubits_and_gates_distribution_params,
    const std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights) {
  this->clear();
  this->qubits_and_gates_distribution_params =
      qubits_and_gates_distribution_params;
  this->gates_weights = gates_weights;
}

unsigned int
QuantumCircuitEnvironment::circuit_invalid_type(FuncOp circuit) const {
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
QuantumCircuitEnvironment::get_circuit_info(FuncOp circuit) {
  if (circuit == nullptr) {
    return {};
  }
  auto [nrQubits, nrGates, depth] = getQubitsInstructionsDepth(circuit);
  return {{"qubits", nrQubits}, {"gates", nrGates}, {"depth", depth}};
}

std::unordered_map<std::string, unsigned int>
QuantumCircuitEnvironment::get_circuit_info() const {
  if (this->circuit_module == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return {};
  }
  return get_circuit_info(FuncOp(this->circuit_module));
}

std::tuple<double, bool> QuantumCircuitEnvironment::step(unsigned int action) {
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

InstructionsTensor<double> QuantumCircuitEnvironment::get_observation() {
  InstructionsTensor<double> observation(this->max_qubits);
  if (this->circuit_module == nullptr) {
    return observation;
  }
  const int NR_QUBITS = getNumberOfQubits(FuncOp(this->circuit_module));
  if (NR_QUBITS == 0) {
    return observation;
  }
  const unsigned int nr_instructions =
      getNumberOfGates(FuncOp(this->circuit_module));
  if (nr_instructions <= 0) {
    // Make N=1 dummy row to allow agent to eat this observation with smaller
    // padding.
    observation.reserve(1);
    observation.append(std::vector(observation.shape[1], 0.0));
    return observation;
  }
  observation.reserve(nr_instructions);

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

    std::vector features(observation.shape[1], 0.0);

    // Controls
    for (int qubit : controls) {
      if (0 <= qubit && qubit < MAX_QUBITS) {
        features[qubit] = -1.0;
      }
    }

    // Targets
    for (int qubit : targets) {
      if (0 <= qubit && qubit < MAX_QUBITS) {
        features[qubit] = 1.0;
      }
    }

    // Gate
    features[GATE_OFFSET + gate_index] = isAdj ? -1.0 : 1.0;

    // params: [angle1, angle2, angle3]
    for (int i = 0; i < MAX_GATE_PARAMS; i++) {
      features[PARAM_OFFSET + i] = params[i];
    }
    observation.append(features);
    instruction_index++;
  });
  return observation;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, bool>
QuantumCircuitEnvironment::getOperatingControlsTargetsParams(Operation *op) {
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