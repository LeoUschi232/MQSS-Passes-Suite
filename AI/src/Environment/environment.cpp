#include "Environment/environment.hpp"

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"
#include "Environment/random_quantum_circuit_generator.hpp"

// MLIR includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

// Cudaq includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

// Utils includes
#include "Support/mlir_utils.hpp"
#include "Utils/info_utils.hpp"
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

QuantumCircuitEnvironment::QuantumCircuitEnvironment(
    unsigned int max_qubits, unsigned int max_steps,
    const fs::path &circuit_path)
    : max_qubits(std::max(GLOBAL_MIN_NR_QUBITS, max_qubits)),
      context_ptr(nullptr), max_steps_per_episode(std::max(1u, max_steps)),
      max_steps_no_improvement(max_steps_per_episode),
      max_steps_no_change(max_steps_per_episode),
      max_steps_same_action(max_steps_per_episode) {
  if (!circuit_path.empty()) {
    this->register_quantum_circuit(circuit_path);
  }
}

QuantumCircuitEnvironment::QuantumCircuitEnvironment(
    unsigned int max_qubits,
    std::unordered_map<std::string, std::string> params)
    : max_qubits(std::max(GLOBAL_MIN_NR_QUBITS, max_qubits)) {
  if (params.find("max_steps_relative_to_qubits") != params.end() &&
      params["max_steps_relative_to_qubits"] == "true") {
    this->max_steps_per_episode = std::max(
        1u, static_cast<unsigned>(max_qubits *
                                  std::stod(params["max_steps_per_episode"])));
    this->max_steps_no_improvement = std::max(
        1u, static_cast<unsigned>(
                max_qubits * std::stod(params["max_steps_no_improvement"])));
    this->max_steps_no_change = std::max(
        1u, static_cast<unsigned>(max_qubits *
                                  std::stod(params["max_steps_no_change"])));
    this->max_steps_same_action = std::max(
        1u, static_cast<unsigned>(max_qubits *
                                  std::stod(params["max_steps_same_action"])));
  } else {
    this->max_steps_per_episode =
        std::max(1ul, std::stoul(params["max_steps_per_episode"]));
    this->max_steps_no_improvement =
        std::max(1ul, std::stoul(params["max_steps_no_improvement"]));
    this->max_steps_no_change =
        std::max(1ul, std::stoul(params["max_steps_no_change"]));
    this->max_steps_same_action =
        std::max(1ul, std::stoul(params["max_steps_same_action"]));
  }
  if (params.find("circuit") != params.end() && !params["circuit"].empty()) {
    this->register_quantum_circuit(params["circuit"]);
  }
}

QuantumCircuitEnvironment::QuantumCircuitEnvironment(
    QuantumCircuitEnvironment &&other) noexcept
    : max_qubits(other.max_qubits),
      circuit_module(std::move(other.circuit_module)),
      context_ptr(std::move(other.context_ptr)),
      circuit_path(std::move(other.circuit_path)),
      max_steps_per_episode(other.max_steps_per_episode),
      max_steps_no_improvement(other.max_steps_no_improvement),
      max_steps_no_change(other.max_steps_no_change),
      max_steps_same_action(other.max_steps_same_action),
      step_per_episode(other.step_per_episode),
      step_no_improvement(other.step_no_improvement),
      step_no_change(other.step_no_change),
      step_same_action(other.step_same_action),
      qubits_cholesky_params(std::move(other.qubits_cholesky_params)),
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
    max_steps_per_episode = other.max_steps_per_episode;
    max_steps_no_improvement = other.max_steps_no_improvement;
    max_steps_no_change = other.max_steps_no_change;
    max_steps_same_action = other.max_steps_same_action;
    step_per_episode = other.step_per_episode;
    step_no_improvement = other.step_no_improvement;
    step_no_change = other.step_no_change;
    step_same_action = other.step_same_action;
    qubits_cholesky_params = std::move(other.qubits_cholesky_params);
    gates_weights = std::move(other.gates_weights);
  }
  return *this;
}

void QuantumCircuitEnvironment::clear(bool hard) {
  if (hard) {
    this->circuit_path = "";
    this->qubits_cholesky_params = std::nullopt;
    this->gates_weights = std::nullopt;
  }
  this->circuit_module = nullptr;
  if (context_ptr && this->context_ptr.get() != nullptr) {
    delete *this->context_ptr.get();
  }
  this->context_ptr = nullptr;
  this->nr_qubits = 0u;
  this->nr_gates = 0u;
  this->depth = 0u;
  this->step_per_episode = 0u;
  this->step_no_improvement = 0u;
  this->step_no_change = 0u;
  this->step_same_action = 0u;
  this->last_action = -1;
  this->terminated = false;
  this->truncated = false;
}

void QuantumCircuitEnvironment::reset() {
  this->clear(/*hard=*/false);
  if (!this->circuit_path.empty()) {
    this->register_quantum_circuit(this->circuit_path);
    return;
  }
  if (qubits_cholesky_params.has_value() && gates_weights.has_value()) {
    auto [nr_qubits, nr_gates, depth, module, context] =
        random_quantum_circuit_from_embedded_statistics(
            qubits_cholesky_params.value(), gates_weights.value(),
            {.max_nr_qubits = static_cast<int>(this->max_qubits),
             .weight_min_multiplier_for_unoccurring_gates = 0.1,
             .probability_additionals_controls = 0.01});
    this->nr_qubits = nr_qubits;
    this->nr_gates = nr_gates;
    this->depth = depth;
    if (this->check() || module == nullptr || context == nullptr) {
      std::cerr << "Failed to randomize a new circuit on reset." << std::endl;
      return;
    }
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
  this->clear(/*hard=*/false);
  this->context_ptr = std::move(context);

  int circuit_validity = validate();
  switch (circuit_validity) {
  case CIRCUIT_VALID:
    break;
  case NO_CIRCUIT:
    std::cerr << "No circuit provided to the environment." << std::endl;
    break;
  case INVALID_NR_QUBITS:
    std::cerr << "Passed circuit has invalid nr qubits." << std::endl;
    break;
  case INVALID_NR_GATES:
    std::cerr << "Passed circuit has invalid nr gates." << std::endl;
    break;
  case INVALID_NR_ALLOCATIONS:
    std::cerr << "Passed circuit has invalid allocations." << std::endl;
    break;
  case AMBIGUOUS_MEASUREMENT:
    std::cerr << "Passed circuit has ambiguous measurements." << std::endl;
    break;
  default:;
  }
  if (circuit_validity != CIRCUIT_VALID) {
    this->clear(/*hard=*/false);
    return false;
  }
  // Register the circuit path and module only if the circuit is valid.
  // Function validate_circuit should already correctly assign nr_qubits,
  // nr_gates and depth.
  this->circuit_path = circuit_path;
  this->circuit_module = circuit;
  return true;
}

bool QuantumCircuitEnvironment::custom_randomize_circuit(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
    const std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights,
    const RandomizerOptions &randomizer_options) {
  try {
    randomizer_options.min_nr_qubits =
        std::max(static_cast<int>(GLOBAL_MIN_NR_QUBITS),
                 randomizer_options.min_nr_qubits);
    randomizer_options.max_nr_qubits = std::min(
        randomizer_options.max_nr_qubits, static_cast<int>(this->max_qubits));

    // Cap nr of qubits but don't cap instructions.
    auto [nr_qubits, nr_gates, depth, module, context] =
        random_quantum_circuit_from_embedded_statistics(
            cholesky_params, gates_weights, randomizer_options);
    this->nr_qubits = nr_qubits;
    this->nr_gates = nr_gates;
    this->depth = depth;
    if (this->check() || module == nullptr || context == nullptr) {
      std::cerr << "Failed to randomize a custom circuit." << std::endl;
      return false;
    }
    this->clear(/*hard=*/false);
    this->circuit_module = module;
    this->context_ptr =
        std::make_unique<MLIRContext *>(std::move(context).release());
  } catch (const std::runtime_error &error) {
    std::cerr << error.what() << std::endl;
    return false;
  }
  return true;
}

void QuantumCircuitEnvironment::register_randomizer_params(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
    const std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights) {
  this->qubits_cholesky_params = cholesky_params;
  this->gates_weights = gates_weights;
}

int QuantumCircuitEnvironment::validate() {
  auto circuit = FuncOp(this->circuit_module);
  this->nr_qubits = 0u;
  this->nr_gates = 0u;
  this->depth = 0u;
  if (circuit == nullptr) {
    return NO_CIRCUIT;
  }
  unsigned int nr_qubits = 0u;
  unsigned int nr_gates = 0u;
  std::vector<unsigned int> depths;
  unsigned int nr_allocations = 0u;
  bool ambiguous_measurement = false;
  circuit.walk([&](Operation *op) -> mlir::WalkResult {
    if (nr_allocations >= 2u || ambiguous_measurement ||
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
      depths.resize(nr_qubits, 0);
      return mlir::WalkResult::advance();
    }
    if (!isGate(op)) {
      return mlir::WalkResult::advance();
    }
    if (isMeasurement(op)) {
      mlir::OperandRange operands = op->getOperands();
      if (operands.size() != 1) {
        ambiguous_measurement = true;
        return mlir::WalkResult::interrupt();
      }
      Value operand = operands.front();
      if (operand.getType().isa<quake::RefType>()) {
        auto qubitIndexOpt =
            extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
        if (!qubitIndexOpt.has_value()) {
          // If the measurement doesn't have a valid qubit indexes, it's
          // ambiguous what the measurement is.
          ambiguous_measurement = true;
          return mlir::WalkResult::interrupt();
        }
        if (int qubitIndex = qubitIndexOpt.value();
            0 <= qubitIndex && qubitIndex < nr_qubits) {
          nr_gates++;
          depths[qubitIndex]++;
        }
        return mlir::WalkResult::advance();
      }
      if (operand.getType().isa<quake::VeqType>()) {
        for (int qubitIndex = 0; qubitIndex < nr_qubits; qubitIndex++) {
          depths[qubitIndex]++;
        }
        nr_gates += operand.getType().dyn_cast<quake::VeqType>().getSize();
        return mlir::WalkResult::advance();
      }
      // If the measurement is neither a RefType nor a VeqType, it's ambiguous
      // what the measurement is.
      ambiguous_measurement = true;
      return mlir::WalkResult::interrupt();
    }
    nr_gates++;
    auto gate = dyn_cast<quake::OperatorInterface>(op);
    std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
    std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
    targets.insert(targets.end(), controls.begin(), controls.end());
    unsigned int max_depth = 0;
    for (int qubit : targets) {
      max_depth = std::max(max_depth, depths[qubit]);
    }
    for (int qubit : targets) {
      depths[qubit] = max_depth + 1;
    }
    return mlir::WalkResult::advance();
  });
  if (nr_qubits < GLOBAL_MIN_NR_QUBITS || nr_qubits > max_qubits) {
    return INVALID_NR_QUBITS;
  }
  if (nr_gates < GLOBAL_MIN_NR_GATES) {
    return INVALID_NR_GATES;
  }
  if (nr_allocations != 1) {
    return INVALID_NR_ALLOCATIONS;
  }
  if (ambiguous_measurement) {
    return AMBIGUOUS_MEASUREMENT;
  }
  this->nr_qubits = nr_qubits;
  this->nr_gates = nr_gates;
  this->depth =
      depths.empty() ? 0u : *std::max_element(depths.begin(), depths.end());
  return CIRCUIT_VALID;
}

std::unordered_map<std::string, unsigned int>
QuantumCircuitEnvironment::get_circuit_info() {
  if (this->circuit_module == nullptr) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return {};
  }
  if (!this->check()) {
    auto [nr_qubits, nr_gates, depth] =
        getQubitsInstructionsDepth(FuncOp(this->circuit_module));
    this->assign(nr_qubits, nr_gates, depth);
  }
  return {{"qubits", this->nr_qubits},
          {"gates", this->nr_gates},
          {"depth", this->depth}};
}

bool QuantumCircuitEnvironment::run_pass(unsigned int pass_index) {
  if (pass_index >= NR_PASSES) {
    return false;
  }
  auto [passname, passptr] = getPassNameAndPointer(pass_index);
  try {
    // Variable context must be a MLIRContext&.
    // Types are:
    // context_ptr = unique_ptr<MLIRContext*>
    // context_ptr.get() = MLIRContext**
    // *context_ptr.get() = MLIRContext*
    // **context_ptr.get() = MLIRContext
    MLIRContext &context = **this->context_ptr.get();
    mlir::PassManager pass_manager(&context);
    pass_manager.addPass(std::move(passptr));

    // If something above throws an exception do not invalidate the circuit but
    // if the pass is going to be attempted, regardless of outcome, invalidate
    // the circuit.
    this->nr_qubits = 0u;
    this->nr_gates = 0u;
    this->depth = 0u;
    if (mlir::failed(pass_manager.run(this->circuit_module))) {
      return false;
    }
  } catch (const std::runtime_error &error) {
    std::cerr << "\nPass " << passname << " failed with " << error.what()
              << std::endl;
    return false;
  }
  return true;
}

/// [Reward, Terminated, Truncated]
std::tuple<double, bool, bool>
QuantumCircuitEnvironment::step(unsigned int action) {
  if (this->terminated || this->truncated) {
    return {0.0, this->terminated, this->truncated};
  }
  if (++this->step_per_episode > this->max_steps_per_episode) {
    this->truncated = true;
    return {0.0, /*Terminated=*/false, /*Truncated=*/true};
  }
  if (this->circuit_module == nullptr) {
    throw std::runtime_error("No circuit registered in the environment.");
  }
  if (action >= NR_PASSES) {
    throw std::runtime_error("Invalid action: " + std::to_string(action));
  }
  double previous_gates;
  double previous_depth;
  if (this->check()) {
    // Assume that if those values are set, they are correct.
    previous_gates = this->nr_gates;
    previous_depth = this->depth;
  } else {
    throw std::runtime_error("QuantumCircuitEnvironment check failed in step.");
  }

  auto [passname, passptr] = getPassNameAndPointer(action);
  try {
    // Variable context must be a MLIRContext&.
    // Types are:
    // context_ptr = unique_ptr<MLIRContext*>
    // context_ptr.get() = MLIRContext**
    // *context_ptr.get() = MLIRContext*
    // **context_ptr.get() = MLIRContext
    MLIRContext &context = **this->context_ptr.get();
    mlir::PassManager pass_manager(&context);
    pass_manager.addPass(std::move(passptr));

    if (mlir::failed(pass_manager.run(this->circuit_module))) {
      throw std::runtime_error("Standard failure.");
    }
  } catch (const std::runtime_error &error) {
    std::cerr << "\nPass " << passname << " failed with " << error.what()
              << std::endl;
    return {0.0, /*Terminated=*/false, /*Truncated=*/false};
  }
  std::unordered_map<std::string, unsigned int> current_circuit_info =
      this->get_circuit_info();

  double current_depth = current_circuit_info["depth"];
  double current_gates = current_circuit_info["gates"];
  double reward =
      previous_depth - current_depth + previous_gates - current_gates;

  if (reward > 0.0) {
    this->step_no_improvement = 0;
  } else if (++this->step_no_improvement > this->max_steps_no_improvement) {
    return {reward, /*Terminated=*/true, /*Truncated=*/false};
  }
  if (previous_circuit_info["depth"] != current_circuit_info["depth"] ||
      previous_circuit_info["gates"] != current_circuit_info["gates"]) {
    // Executing the same action many times in a row is only a valid termination
    // criterion IFF that action does not change the circuit.
    this->step_no_change = 0;
    this->step_same_action = 0;
  } else if (++this->step_no_change > this->max_steps_no_change) {
    this->terminated = true;
    return {reward, /*Terminated=*/true, /*Truncated=*/false};
  } else if (static_cast<int>(action) != this->last_action) {
    this->step_same_action = 0;
  } else if (++this->step_same_action > this->max_steps_same_action) {
    this->terminated = true;
    return {reward, /*Terminated=*/true, /*Truncated=*/false};
  }
  this->last_action = static_cast<int>(action);
  assert(!this->terminated || !this->truncated);
  return {reward, /*Terminated=*/false, /*Truncated=*/false};
}

InstructionsTensor<double> QuantumCircuitEnvironment::get_observation() {
  InstructionsTensor<double> observation(this->max_qubits);
  observation.reserve(/*nr_instructions=*/GLOBAL_MIN_NR_GATES);
  if (this->circuit_module == nullptr || this->truncated) {
    // Changed to exclude this->truncated so that for truncated episodes, we
    // return the actual observation for bootstrapping.
    observation.pad(/*toNrInstructions=*/GLOBAL_MIN_NR_GATES, /*value=*/0.0);
    return observation;
  }
  const unsigned int nr_instructions =
      getNumberOfGates(FuncOp(this->circuit_module));
  if (nr_instructions < GLOBAL_MIN_NR_GATES) {
    // Any valid normal circuit should have at least 2 instructions.
    observation.pad(/*toNrInstructions=*/GLOBAL_MIN_NR_GATES, /*value=*/0.0);
    return observation;
  }
  observation.reserve(nr_instructions);

  const int MAX_QUBITS = this->max_qubits;
  const int GATE_OFFSET = MAX_QUBITS;
  const int PARAM_OFFSET = GATE_OFFSET + NR_GATES;

  int instruction_index = 0;
  this->circuit_module.walk([&](Operation *op) {
    if (!isGate(op)) {
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

    if (isMeasurement(op)) {
      targets = getMeasurementTargets(op, nr_qubits);
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

void QuantumCircuitEnvironment::assign(unsigned int nr_qubits,
                                       unsigned int nr_gates,
                                       unsigned int depth) {
  this->nr_qubits = nr_qubits;
  this->nr_gates = nr_gates;
  this->depth = depth;
  if (!this->check()) {
    this->nr_qubits = 0u;
    this->nr_gates = 0u;
    this->depth = 0u;
  }
}
bool QuantumCircuitEnvironment::check() const {
  return this->nr_qubits >= GLOBAL_MIN_NR_QUBITS &&
         this->nr_qubits <= this->max_qubits &&
         this->nr_gates >= GLOBAL_MIN_NR_GATES && this->depth > 0u;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, bool>
QuantumCircuitEnvironment::getOperatingControlsTargetsParams(Operation *op) {
  if (isMeasurement(op) || !isGate(op)) {
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