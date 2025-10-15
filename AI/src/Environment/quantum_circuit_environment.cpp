#include "Environment/quantum_circuit_environment.hpp"

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"
#include "Environment/random_quantum_circuit_generator.hpp"

// MLIR includes
#include "mlir/Dialect/Func/IR/FuncOps.h"

// Cudaq includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

// Utils includes
#include "Support/mlir_utils.hpp"
#include "Utils/info_utils.hpp"
#include "Utils/passes_utils.hpp"
#include "Utils/tensor_utils.hpp"

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
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

QuantumCircuitEnvironment::QuantumCircuitEnvironment(
    unsigned int max_qubits, unsigned int max_steps,
    const fs::path &circuit_path)
    : max_qubits(std::max(GLOBAL_MIN_NR_QUBITS, max_qubits)),
      max_steps_per_episode(std::max(1u, max_steps)),
      max_steps_no_improvement(max_steps_per_episode),
      max_steps_no_change(max_steps_per_episode),
      max_steps_same_action(max_steps_per_episode) {
  if (!circuit_path.empty()) {
    this->register_quantum_circuit(circuit_path);
  }
}

QuantumCircuitEnvironment::QuantumCircuitEnvironment(unsigned int max_qubits)
    : max_qubits(std::max(GLOBAL_MIN_NR_QUBITS, max_qubits)) {
  this->device = GLOBAL_PARAMS["device"].to_device_type();
  this->max_steps_per_episode = std::max(
      static_cast<unsigned>(GLOBAL_PARAMS["max_steps_per_episode"].to_int()),
      MIN_NR_STEPS);
  this->max_steps_no_improvement = std::max(
      static_cast<unsigned>(GLOBAL_PARAMS["max_steps_no_improvement"].to_int()),
      MIN_NR_STEPS);
  this->max_steps_no_change = std::max(
      static_cast<unsigned>(GLOBAL_PARAMS["max_steps_no_change"].to_int()),
      MIN_NR_STEPS);
  this->max_steps_same_action = std::max(
      static_cast<unsigned>(GLOBAL_PARAMS["max_steps_same_action"].to_int()),
      MIN_NR_STEPS);
  // Do not worry about not having a circuit because the method
  // register_quantum_circuit will handle empty strings.
  this->register_quantum_circuit(GLOBAL_PARAMS["circuit"].to_string());
}

void QuantumCircuitEnvironment::clear(bool hard) {
  if (hard) {
    this->circuit_path = "";
    this->qubits_cholesky_params = std::nullopt;
    this->gates_weights = std::nullopt;
  }
  this->circuit.clear();
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
    // The QuantumCircuit object validates itself on construction, so no need
    // for extra validation.
    this->circuit = random_quantum_circuit_from_embedded_statistics(
        qubits_cholesky_params.value(), gates_weights.value(),
        {.max_nr_qubits = static_cast<int>(this->max_qubits),
         .weight_min_multiplier_for_unoccurring_gates = 0.1,
         .probability_additionals_controls = 0.01});
    this->validate();
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
  this->circuit = QuantumCircuit(circuit_path);
  int circuit_validity = this->get_advanced_circuit_validity();
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
    // Cap nr of qubits but don't cap nr of gates.
    randomizer_options.max_nr_qubits = std::min(
        randomizer_options.max_nr_qubits, static_cast<int>(this->max_qubits));
    this->circuit = random_quantum_circuit_from_embedded_statistics(
        cholesky_params, gates_weights, randomizer_options);
  } catch (const std::runtime_error &error) {
    std::cerr << error.what() << std::endl;
    return false;
  }
  return this->validate();
}

void QuantumCircuitEnvironment::register_randomizer_params(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
    const std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights) {
  this->qubits_cholesky_params = cholesky_params;
  this->gates_weights = gates_weights;
}

int QuantumCircuitEnvironment::get_advanced_circuit_validity() {
  if (!this->circuit.exists()) {
    return NO_CIRCUIT;
  }
  unsigned int nr_qubits = 0u;
  unsigned int nr_gates = 0u;
  std::vector<unsigned int> depths;
  unsigned int nr_allocations = 0u;
  bool ambiguous_measurement = false;
  FuncOp(this->circuit).walk([&](Operation *op) -> mlir::WalkResult {
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
  if (nr_qubits != this->circuit.get_nr_qubits() ||
      nr_gates != this->circuit.get_nr_gates()) {
    this->circuit.recompute();
  }
  if (nr_qubits != this->circuit.get_nr_qubits() ||
      nr_gates != this->circuit.get_nr_gates()) {
    throw std::runtime_error(
        "Mismatch in algorithms computing nr_circuits and nr_gates between "
        "QuantumCircuit and QuantumCircuitEnvironment.");
  }
  return CIRCUIT_VALID;
}

std::unordered_map<std::string, unsigned int>
QuantumCircuitEnvironment::get_circuit_info() const {
  if (!this->circuit.exists()) {
    std::cerr << "No circuit registered in the environment." << std::endl;
    return {};
  }
  return {{"qubits", this->circuit.get_nr_qubits()},
          {"gates", this->circuit.get_nr_gates()},
          {"depth", this->circuit.get_depth()}};
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
  if (!this->circuit.exists()) {
    throw std::runtime_error("No circuit registered in the environment.");
  }
  if (action >= NR_PASSES) {
    throw std::runtime_error("Invalid action: " + std::to_string(action));
  }
  double previous_nr_gates = this->circuit.get_nr_gates();
  double previous_depth = this->circuit.get_depth();
  if (!this->circuit.run_pass(/*pass_index=*/action)) {
    std::cerr << "Action " << std::to_string(action) << " failed." << std::endl;
    return {0.0, /*Terminated=*/false, /*Truncated=*/false};
  }
  double nr_gates_reduction = previous_nr_gates - this->circuit.get_nr_gates();
  double depth_reduction = previous_depth - this->circuit.get_depth();
  double reward = nr_gates_reduction + depth_reduction;

  if (reward > 0.0) {
    this->step_no_improvement = 0;
  } else if (++this->step_no_improvement > this->max_steps_no_improvement) {
    return {reward, /*Terminated=*/true, /*Truncated=*/false};
  }
  if (!isclose(nr_gates_reduction, 0.0) && !isclose(depth_reduction, 0.0)) {
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

InstructionsTensor<float> QuantumCircuitEnvironment::get_observation() const {
  InstructionsTensor<float> observation(this->max_qubits);
  observation.reserve(/*nr_instructions=*/GLOBAL_MIN_NR_GATES);
  if (!this->circuit.exists() || this->truncated) {
    // Changed to exclude this->truncated so that for truncated episodes, we
    // return the actual observation for bootstrapping.
    observation.pad(/*toNrInstructions=*/GLOBAL_MIN_NR_GATES, /*value=*/0.0f);
    return observation;
  }
  const unsigned int nr_gates = this->circuit.get_nr_gates();
  const unsigned int nr_qubits = this->circuit.get_nr_qubits();
  if (nr_gates < GLOBAL_MIN_NR_GATES || nr_qubits < GLOBAL_MIN_NR_QUBITS) {
    // Any valid normal circuit should have at least 2 instructions.
    observation.pad(/*toNrInstructions=*/GLOBAL_MIN_NR_GATES, /*value=*/0.0f);
    return observation;
  }
  observation.reserve(nr_gates);

  const int MAX_QUBITS = this->max_qubits;
  const int GATE_OFFSET = MAX_QUBITS;
  const int PARAM_OFFSET = GATE_OFFSET + NR_GATES;
  int instruction_index = 0;
  FuncOp(this->circuit).walk([&](Operation *op) {
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
    std::vector float_params(MAX_GATE_PARAMS, 0.0f);
    bool isAdj = false;

    if (isMeasurement(op)) {
      targets = getMeasurementTargets(op, nr_qubits);
    } else {
      std::vector double_params(MAX_GATE_PARAMS, 0.0);
      std::tie(controls, targets, double_params, isAdj) =
          getOperatingControlsTargetsParams(op);
      for (unsigned i = 0u; i < MAX_GATE_PARAMS; i++) {
        float_params[i] = static_cast<float>(double_params[i]);
      }
    }
    std::vector features(observation.shape[1], 0.0f);

    // Controls
    for (int qubit : controls) {
      if (0 <= qubit && qubit < MAX_QUBITS) {
        features[qubit] = -1.0f;
      }
    }

    // Targets
    for (int qubit : targets) {
      if (0 <= qubit && qubit < MAX_QUBITS) {
        features[qubit] = 1.0f;
      }
    }

    // Gate
    features[GATE_OFFSET + gate_index] = isAdj ? -1.0f : 1.0f;

    // params: [angle1, angle2, angle3]
    for (int i = 0; i < MAX_GATE_PARAMS; i++) {
      features[PARAM_OFFSET + i] = float_params[i];
    }
    observation.append(features);
    instruction_index++;
  });
  return observation;
}

torch::Tensor QuantumCircuitEnvironment::get_observation_as_torch_tensor(
    std::optional<torch::TensorOptions> tensor_options) const {
  torch::TensorOptions options = tensor_options.value_or(
      torch::TensorOptions().dtype(torch::kFloat32).device(this->device));
  InstructionsTensor<float> observation = this->get_observation();
  if (GLOBAL_PARAMS["print_diagnostics"].to_bool()) {
    check_tensor(observation);
  }
  unsigned int N = observation.shape[0];
  unsigned int IRS = observation.shape[1];
  if (N < GLOBAL_MIN_NR_GATES || IRS < MIN_IRS) {
    return torch::zeros({GLOBAL_MIN_NR_GATES, std::max(IRS, MIN_IRS)}, options);
  }
  return torch::from_blob(observation.raw(), {N, IRS}, options).clone();
}

bool QuantumCircuitEnvironment::validate() {
  if (this->circuit.get_nr_qubits() > this->max_qubits ||
      !this->circuit.validate()) {
    this->clear(/*hard=*/false);
    return false;
  }
  return true;
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