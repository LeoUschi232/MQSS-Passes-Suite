#include "Environment/random_quantum_circuit_generator.hpp"

// Support includes
#include "Support/mlir_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
/// The usages of llvm functions must come before the QuakeOps header which
/// expects them.
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
////////////////////////////////////////////////////////////////////////////////

// Cudaq includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

// Utils includes
#include "Utils/info_utils.hpp"
#include "Utils/tensor_utils.hpp"

// Yaml include
#include <yaml-cpp/yaml.h>

// Standard library includes
#include <filesystem>
#include <iostream>
#include <random>
#include <unordered_set>

namespace fs = std::filesystem;
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {

static GateSpec gateSpecFromIndex(unsigned int idx) {
  if (idx >= OPERATIONS_SUBSET_SIZE) {
    llvm::report_fatal_error("gateSpecFromIndex: out of range");
  }
  switch (idx) {
  case X_INDEX:
    return {X, false, 0, 0, false, false};
  case CX_INDEX:
    return {X, false, 1, 0, false, false};
  case CCX_INDEX:
    return {X, false, 2, 0, false, false};
  case C3plus_X_INDEX:
    return {X, false, -1, 3, true, false};
  case Y_INDEX:
    return {Y, false, 0, 0, false, false};
  case controlled_Y_INDEX:
    return {Y, false, -1, 1, true, false};
  case Z_INDEX:
    return {Z, false, 0, 0, false, false};
  case controlled_Z_INDEX:
    return {Z, false, -1, 1, true, false};
  case H_INDEX:
    return {H, false, 0, 0, false, false};
  case controlled_H_INDEX:
    return {H, false, -1, 1, true, false};
  case S_INDEX:
    return {S, false, 0, 0, false, false};
  case controlled_S_INDEX:
    return {S, false, -1, 1, true, false};
  case SDG_INDEX:
    return {S, true, 0, 0, false, false};
  case controlled_SDG_INDEX:
    return {S, true, -1, 1, true, false};
  case T_INDEX:
    return {T, false, 0, 0, false, false};
  case controlled_T_INDEX:
    return {T, false, -1, 1, true, false};
  case TDG_INDEX:
    return {T, true, 0, 0, false, false};
  case controlled_TDG_INDEX:
    return {T, true, -1, 1, true, false};
  case RX_INDEX:
    return {RX, false, 0, 0, false, false};
  case controlled_RX_INDEX:
    return {RX, false, -1, 1, true, false};
  case RY_INDEX:
    return {RY, false, 0, 0, false, false};
  case controlled_RY_INDEX:
    return {RY, false, -1, 1, true, false};
  case RZ_INDEX:
    return {RZ, false, 0, 0, false, false};
  case controlled_RZ_INDEX:
    return {RZ, false, -1, 1, true, false};
  case SWAP_INDEX:
    return {SWAP, false, 0, 0, false, true};
  case controlled_SWAP_INDEX:
    return {SWAP, false, -1, 1, true, true};
  case R1_INDEX:
    return {R1, false, 0, 0, false, false};
  case controlled_R1_INDEX:
    return {R1, false, -1, 1, true, false};
  case U2_INDEX:
    return {U2, false, 0, 0, false, false};
  case controlled_U2_INDEX:
    return {U2, false, -1, 1, true, false};
  case U3_INDEX:
    return {U3, false, 0, 0, false, false};
  case controlled_U3_INDEX:
    return {U3, false, -1, 1, true, false};
  case PHASED_RX_INDEX:
    return {PHASED_RX, false, 0, 0, false, false};
  case controlled_PHASED_RX_INDEX:
    return {PHASED_RX, false, -1, 1, true, false};
  default:
    std::cerr << "gateSpecFromIndex: unknown index " << idx << std::endl;
  }
  return {0, false, 0, 0, false, false};
}

std::pair<std::vector<int>, std::vector<int>> sampleDistinctTargetsAndControls(
    unsigned int nr_targets, unsigned int nr_controls, unsigned int nr_qubits) {
  if (nr_targets + nr_controls > nr_qubits) {
    throw std::runtime_error(
        "sampleDistinctTargetsAndControls: not enough qubits");
  }
  std::vector<int> targets;
  targets.reserve(nr_targets);
  std::vector<int> controls;
  controls.reserve(nr_controls);
  std::unordered_set<unsigned int> used{};
  std::uniform_int_distribution qubit_distribution(0u, nr_qubits - 1);
  for (unsigned int i = 0; i < nr_targets; i++) {
    unsigned int qubit = qubit_distribution(qc_rng());
    while (!used.insert(qubit).second) {
      // On collisions instead of resampling to infinity, just keep trying the
      // next qubit in the cycle until we find a free one. This is certain to
      // terminate in O(nr_qubits) time whereas resampling again has a
      // probability of repikcing used qubits more often.
      qubit = (qubit + 1) % nr_qubits;
    }
    targets.push_back(qubit);
  }
  for (unsigned int i = 0; i < nr_controls; i++) {
    unsigned int qubit = qubit_distribution(qc_rng());
    while (!used.insert(qubit).second) {
      qubit = (qubit + 1) % nr_qubits;
    }
    controls.push_back(qubit);
  }
  if (targets.size() != nr_targets || controls.size() != nr_controls) {
    throw std::runtime_error("sampleDistinctTargetsAndControls: targets or "
                             "controls didn't aquire desired sizes.");
  }
  return {targets, controls};
}

std::vector<double> makeAngles(int baseGate) {
  switch (baseGate) {
  case RX:
  case RY:
  case RZ:
  case R1:
    return {randomAngle()};
  case U2:
    return {randomAngle(), randomAngle()};
  case U3:
    return {randomAngle(), randomAngle(), randomAngle()};
  case PHASED_RX:
    return {randomAngle(), randomAngle()};
  default:
    return {};
  }
}

std::pair<unsigned int, unsigned int> sample_nr_qubits_and_gates_from_cholesky(
    std::tuple<double, double, double, double, double>
        qubits_and_gates_distribution_params) {
  auto [mean_qubits, mean_gates, L11, L21, L22] =
      qubits_and_gates_distribution_params;
  double nr_qubits = 0.0;
  double nr_gates = 0.0;
  while (nr_qubits < 2.0 || nr_gates < 2.0) {
    std::normal_distribution ndist(0.0, 1.0);
    double z1 = ndist(qc_rng()), z2 = ndist(qc_rng());
    nr_qubits = mean_qubits + L11 * z1;
    nr_gates = mean_gates + L21 * z1 + L22 * z2;
  }
  return {std::round(nr_qubits), std::round(nr_gates)};
}

std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
random_quantum_circuit_from_embedded_statistics(
    const std::tuple<double, double, double, double, double>
        &qubits_and_gates_distribution_params,
    std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights,
    const RandomizerOptions &randomizer_options) {
  double multiplier =
      randomizer_options.weight_min_multiplier_for_unoccurring_gates;
  if (multiplier > 0.0) {
    unsigned int minimum_weight = std::numeric_limits<unsigned int>::max();
    for (const auto &weight : gates_weights) {
      if (weight > 0 && weight < minimum_weight) {
        minimum_weight = weight;
      }
    }
    minimum_weight = std::round(multiplier * minimum_weight);
    if (minimum_weight > 0u) {
      for (unsigned int i = 0; i < gates_weights.size(); i++) {
        if (gates_weights[i] <= 0) {
          gates_weights[i] = minimum_weight;
        }
      }
    }
  }
  auto [nr_qubits, nr_gates] = sample_nr_qubits_and_gates_from_cholesky(
      qubits_and_gates_distribution_params);
  // At least 2 qubits and more gates than qubits.
  // DO NOT Compare unsigned int to int
  if (randomizer_options.min_nr_qubits >= 2) {
    nr_qubits = std::max(
        static_cast<unsigned>(randomizer_options.min_nr_qubits), nr_qubits);
  }
  if (randomizer_options.max_nr_qubits >= 2) {
    nr_qubits = std::min(
        static_cast<unsigned>(randomizer_options.max_nr_qubits), nr_qubits);
  }
  if (randomizer_options.exact_nr_qubits >= 2) {
    nr_qubits = static_cast<unsigned>(randomizer_options.exact_nr_qubits);
  }
  // Whatever the randomizer options, the nr of qubits must be at least 2.
  nr_qubits = std::max(2u, nr_qubits);
  if (randomizer_options.min_nr_gates >= 2) {
    nr_gates = std::max(static_cast<unsigned>(randomizer_options.min_nr_gates),
                        nr_gates);
  }
  if (randomizer_options.max_nr_gates >= 2) {
    nr_gates = std::min(static_cast<unsigned>(randomizer_options.max_nr_gates),
                        nr_gates);
  }
  if (randomizer_options.exact_nr_gates >= 2) {
    nr_gates = static_cast<unsigned>(randomizer_options.exact_nr_gates);
  }
  // Whatever the randomizer options, the nr of gates must be at least 2
  nr_gates = std::max(nr_gates, nr_qubits + 1);

  auto buildSetup = beginReconstruction("__nvqpp__mlirgen__Random", nr_qubits);

  // Sample a gate according to operations-only subset of gates_weights and
  // later according to measurements-only subset of gates_weights.
  std::discrete_distribution<size_t> operations_distribution(
      gates_weights.begin(), gates_weights.begin() + OPERATIONS_SUBSET_SIZE);
  std::discrete_distribution<size_t> measurements_distribution(
      gates_weights.begin() + OPERATIONS_SUBSET_SIZE, gates_weights.end());

  for (unsigned int i = 0; i < nr_gates - nr_qubits; i++) {
    const unsigned int idx = operations_distribution(qc_rng());
    const auto [baseGate, isAdj, exactControls, minControls, allowExtraControls,
                isSwap] = gateSpecFromIndex(idx);

    unsigned int nr_targets = isSwap ? 2 : 1;
    unsigned int nr_controls = 0;
    // Regardless of what exactControls and minControls are, we cannot be using
    // more qubits than are available for use.
    if (exactControls >= 0) {
      nr_controls = std::min(static_cast<unsigned>(exactControls),
                             nr_qubits - nr_targets);
    } else if (minControls > 0) {
      nr_controls =
          std::min(static_cast<unsigned>(minControls), nr_qubits - nr_targets);
    }
    if (exactControls < 0 && allowExtraControls &&
        probability_additionals_controls > 0.0) {
      while (nr_controls + nr_targets < nr_qubits &&
             random01() < probability_additionals_controls) {
        nr_controls++;
      }
    }
    auto [targets, controls] =
        sampleDistinctTargetsAndControls(nr_targets, nr_controls, nr_qubits);
    // Angles if any and emit the operation.
    std::vector<double> angles = makeAngles(baseGate);
    insertGate(buildSetup, baseGate, targets, controls, angles, isAdj);
  }
  for (unsigned int qubit = 0; qubit < nr_qubits; qubit++) {
    std::vector targets{static_cast<int>(qubit)};
    if (const unsigned int idx =
            measurements_distribution(qc_rng()) + OPERATIONS_SUBSET_SIZE;
        idx == MX_INDEX) {
      insertGate(buildSetup, MX, targets);
    } else if (idx == MY_INDEX) {
      insertGate(buildSetup, MY, targets);
    } else if (idx == MZ_INDEX) {
      insertGate(buildSetup, MZ, targets);
    } else {
      throw std::runtime_error("Unknown measurement index " +
                               std::to_string(idx));
    }
  }
  return {buildSetup.module, std::move(buildSetup.ctxOwner)};
}

std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
random_quantum_circuit_from_yaml_statistics(
    const fs::path &statistics_yaml_file_path,
    const std::pair<int, int> &cap_nr_qubits,
    const std::pair<int, int> &cap_nr_gates,
    bool give_small_probability_to_unoccurring_gates,
    double probability_additionals_qubits) {
  if (!fs::exists(statistics_yaml_file_path) ||
      !fs::is_regular_file(statistics_yaml_file_path)) {
    std::cerr << "File: " << statistics_yaml_file_path
              << " does not exist or is not a regular file." << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////
  // Load from YAML
  YAML::Node statistics = YAML::LoadFile(statistics_yaml_file_path.string());

  auto params = statistics["qubits_and_gates_distribution_params"];
  if (!params || !statistics["gates_weights"]) {
    throw std::runtime_error("YAML missing required sections.");
  }

  auto get_double = [](const YAML::Node &node, const char *key) -> double {
    if (!node[key]) {
      throw std::runtime_error(std::string("Missing key: ") + key);
    }
    return node[key].as<double>();
  };
  auto get_unsigned = [](const YAML::Node &node,
                         const char *key) -> unsigned int {
    if (!node[key]) {
      return 0u;
    }
    return node[key].as<unsigned int>();
  };

  std::tuple qubits_and_gates_distribution_params = {
      get_double(params, "mean_qubits"), get_double(params, "mean_gates"),
      get_double(params, "cholesky_L11"), get_double(params, "cholesky_L21"),
      get_double(params, "cholesky_L22")};

  std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights{};
  const YAML::Node &yaml_gates_weights = statistics["gates_weights"];

  gates_weights[X_INDEX] = get_unsigned(yaml_gates_weights, "X");
  gates_weights[CX_INDEX] = get_unsigned(yaml_gates_weights, "CX");
  gates_weights[CCX_INDEX] = get_unsigned(yaml_gates_weights, "CCX");
  gates_weights[C3plus_X_INDEX] = get_unsigned(yaml_gates_weights, "C3plus_X");
  gates_weights[Y_INDEX] = get_unsigned(yaml_gates_weights, "Y");
  gates_weights[controlled_Y_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_Y");
  gates_weights[Z_INDEX] = get_unsigned(yaml_gates_weights, "Z");
  gates_weights[controlled_Z_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_Z");
  gates_weights[H_INDEX] = get_unsigned(yaml_gates_weights, "H");
  gates_weights[controlled_H_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_H");
  gates_weights[S_INDEX] = get_unsigned(yaml_gates_weights, "S");
  gates_weights[controlled_S_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_S");
  gates_weights[SDG_INDEX] = get_unsigned(yaml_gates_weights, "SDG");
  gates_weights[controlled_SDG_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_SDG");
  gates_weights[T_INDEX] = get_unsigned(yaml_gates_weights, "T");
  gates_weights[controlled_T_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_T");
  gates_weights[TDG_INDEX] = get_unsigned(yaml_gates_weights, "TDG");
  gates_weights[controlled_TDG_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_TDG");
  gates_weights[RX_INDEX] = get_unsigned(yaml_gates_weights, "RX");
  gates_weights[controlled_RX_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_RX");
  gates_weights[RY_INDEX] = get_unsigned(yaml_gates_weights, "RY");
  gates_weights[controlled_RY_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_RY");
  gates_weights[RZ_INDEX] = get_unsigned(yaml_gates_weights, "RZ");
  gates_weights[controlled_RZ_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_RZ");
  gates_weights[SWAP_INDEX] = get_unsigned(yaml_gates_weights, "SWAP");
  gates_weights[controlled_SWAP_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_SWAP");
  gates_weights[R1_INDEX] = get_unsigned(yaml_gates_weights, "R1");
  gates_weights[controlled_R1_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_R1");
  gates_weights[U2_INDEX] = get_unsigned(yaml_gates_weights, "U2");
  gates_weights[controlled_U2_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_U2");
  gates_weights[U3_INDEX] = get_unsigned(yaml_gates_weights, "U3");
  gates_weights[controlled_U3_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_U3");
  gates_weights[PHASED_RX_INDEX] =
      get_unsigned(yaml_gates_weights, "PHASED_RX");
  gates_weights[controlled_PHASED_RX_INDEX] =
      get_unsigned(yaml_gates_weights, "controlled_PHASED_RX");
  gates_weights[MX_INDEX] = get_unsigned(yaml_gates_weights, "MX");
  gates_weights[MY_INDEX] = get_unsigned(yaml_gates_weights, "MY");
  gates_weights[MZ_INDEX] = get_unsigned(yaml_gates_weights, "MZ");

  return random_quantum_circuit_from_embedded_statistics(
      qubits_and_gates_distribution_params, gates_weights, cap_nr_qubits,
      cap_nr_gates, give_small_probability_to_unoccurring_gates,
      probability_additionals_qubits);
}

} // namespace ai_pass_selector
