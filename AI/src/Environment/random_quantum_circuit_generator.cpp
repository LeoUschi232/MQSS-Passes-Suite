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
  if (idx >= GATES_WEIGHTS_SIZE) {
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
  case MX_INDEX:
    return {MX, false, 0, 0, false, false};
  case MY_INDEX:
    return {MY, false, 0, 0, false, false};
  case MZ_INDEX:
    return {MZ, false, 0, 0, false, false};
  default:
    std::cerr << "gateSpecFromIndex: unknown index " << idx << std::endl;
  }
  return {0, false, 0, 0, false, false};
}

std::pair<std::vector<int>, std::vector<int>>
sample_distinct_targets_and_controls(unsigned int nr_targets,
                                     unsigned int nr_controls,
                                     unsigned int nr_qubits) {
  if (nr_targets + nr_controls > nr_qubits) {
    throw std::runtime_error("nr_targets = " + std::to_string(nr_targets) +
                             "\nnr_controls = " + std::to_string(nr_controls) +
                             "\nnr_targets+nr_controls = " +
                             std::to_string(nr_targets + nr_controls) +
                             "\nnr_qubits = " + std::to_string(nr_qubits));
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
    throw std::runtime_error(
        "targets.size() = " + std::to_string(targets.size()) +
        "\nnr_targets = " + std::to_string(nr_targets) +
        "\ncontrols.size() = " + std::to_string(controls.size()) +
        "\nnr_controls = " + std::to_string(nr_controls));
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

std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>
sample_nr_qubits_gates_operations_measurements(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params) {
  auto [mean_qubits, mean_gates, mean_operations, mean_measurements, qubits_L11,
        gates_L21, gates_L22, operations_L21, operations_L22, measurements_L21,
        measurements_L22] = cholesky_params;
  double nr_qubits = 0.0;
  double nr_gates = 0.0;
  double nr_operations = 0.0;
  double nr_measurements = 0.0;
  std::normal_distribution normal_distribution(0.0, 1.0);
  double z1 = 0.0;
  while (nr_qubits < GLOBAL_MIN_NR_QUBITS) {
    z1 = normal_distribution(qc_rng());
    nr_qubits = mean_qubits + qubits_L11 * z1;
  }
  while (nr_gates < GLOBAL_MIN_NR_GATES || nr_operations < 0.0 ||
         nr_measurements < 0.0) {
    double z2 = normal_distribution(qc_rng());
    nr_gates = mean_gates + gates_L21 * z1 + gates_L22 * z2;
    nr_operations = mean_operations + operations_L21 * z1 + operations_L22 * z2;
    nr_measurements =
        mean_measurements + measurements_L21 * z1 + measurements_L22 * z2;
  }
  return {std::round(nr_qubits), std::round(nr_gates),
          std::round(nr_operations), std::round(nr_measurements)};
}

std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>
get_nr_qubits_gates_operations_measurements(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
    const RandomizerOptions &randomizer_options) {
  unsigned int nr_qubits = 0u, nr_gates = 0u, nr_operations = 0u;
  if (randomizer_options.exact_nr_qubits >=
          static_cast<int>(GLOBAL_MIN_NR_QUBITS) &&
      randomizer_options.exact_nr_gates >=
          static_cast<int>(GLOBAL_MIN_NR_GATES) &&
      randomizer_options.exact_nr_operations >= 0u &&
      randomizer_options.exact_nr_operations <=
          randomizer_options.exact_nr_gates) {
    nr_qubits = static_cast<unsigned>(randomizer_options.exact_nr_qubits);
    nr_gates = static_cast<unsigned>(randomizer_options.exact_nr_gates);
    nr_operations =
        static_cast<unsigned>(randomizer_options.exact_nr_operations);
  } else {
    // If not all exact parameters are set OR they are inconsistent, ignore them
    // all completely.
    // Number of measurements will be inferred from nr_gates and nr_operations.
    unsigned int _;
    std::tie(nr_qubits, nr_gates, nr_operations, _) =
        sample_nr_qubits_gates_operations_measurements(cholesky_params);

    // DO NOT Compare unsigned int to int
    if (randomizer_options.min_nr_qubits >=
            static_cast<int>(GLOBAL_MIN_NR_QUBITS) &&
        randomizer_options.min_nr_qubits > static_cast<int>(nr_qubits)) {
      nr_qubits = static_cast<unsigned>(randomizer_options.min_nr_qubits);
    }
    if (randomizer_options.max_nr_qubits >=
            static_cast<int>(GLOBAL_MIN_NR_QUBITS) &&
        randomizer_options.max_nr_qubits < static_cast<int>(nr_qubits)) {
      nr_qubits = static_cast<unsigned>(randomizer_options.max_nr_qubits);
    }
    // Whatever the randomizer options, the nr of qubits must be at least 2.
    nr_qubits = std::max(GLOBAL_MIN_NR_QUBITS, nr_qubits);
    if (randomizer_options.min_nr_gates >=
            static_cast<int>(GLOBAL_MIN_NR_GATES) &&
        randomizer_options.min_nr_gates > static_cast<int>(nr_gates)) {
      nr_gates = static_cast<unsigned>(randomizer_options.min_nr_gates);
    }
    if (randomizer_options.max_nr_gates >=
            static_cast<int>(GLOBAL_MIN_NR_GATES) &&
        randomizer_options.max_nr_gates < static_cast<int>(nr_gates)) {
      nr_gates = static_cast<unsigned>(randomizer_options.max_nr_gates);
    }
    // Whatever the randomizer options, the nr_gates must be at least 2.
    nr_gates = std::max(GLOBAL_MIN_NR_GATES, nr_gates);
    if (randomizer_options.min_nr_operations >
        static_cast<int>(nr_operations)) {
      nr_operations =
          static_cast<unsigned>(randomizer_options.min_nr_operations);
    }
    if (randomizer_options.max_nr_operations <
        static_cast<int>(nr_operations)) {
      nr_operations =
          static_cast<unsigned>(randomizer_options.max_nr_operations);
    }
  }
  // Make nr_gates bind stronger than nr_operations.
  if (nr_operations > nr_gates) {
    nr_operations = nr_gates;
  }
  return {nr_qubits, nr_gates, nr_operations,
          /*nr_measurements=*/nr_gates - nr_operations};
}

void adjust_gates_weights(
    double multiplier, unsigned int subset_size,
    std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights) {
  if (multiplier > 0.0) {
    unsigned int minimum_weight = std::numeric_limits<unsigned int>::max();
    for (unsigned int i = 0; i < subset_size; i++) {
      if (gates_weights[i] > 0 && gates_weights[i] < minimum_weight) {
        minimum_weight = gates_weights[i];
      }
    }
    minimum_weight = std::round(multiplier * minimum_weight);
    if (minimum_weight > 0u) {
      for (unsigned int i = 0; i < subset_size; i++) {
        if (gates_weights[i] <= 0) {
          gates_weights[i] = minimum_weight;
        }
      }
    }
  }
}

QuantumCircuit random_quantum_circuit_from_embedded_statistics(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
    std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights,
    const RandomizerOptions &randomizer_options) {
  if (randomizer_options.seed.has_value()) {
    seed_qc_rng(randomizer_options.seed.value());
  }
  unsigned int subset_size = randomizer_options.allow_measurements_as_gates
                                 ? GATES_WEIGHTS_SIZE
                                 : OPERATIONS_SUBSET_SIZE;
  adjust_gates_weights(
      randomizer_options.weight_min_multiplier_for_unoccurring_gates,
      subset_size, gates_weights);
  auto [nr_qubits, nr_gates, nr_operations, nr_measurements] =
      get_nr_qubits_gates_operations_measurements(cholesky_params,
                                                  randomizer_options);
  auto buildSetup = beginQuantumCircuitConstruction(
      /*kernel_name=*/"__nvqpp__mlirgen__Random", nr_qubits);
  std::vector<unsigned int> depths(nr_qubits, 0);

  // Sample a gate according to operations-only subset of gates_weights and
  // later according to measurements-only subset of gates_weights.
  std::discrete_distribution<size_t> operations_distribution(
      gates_weights.begin(), gates_weights.begin() + subset_size);
  std::discrete_distribution<size_t> measurements_distribution(
      gates_weights.begin() + OPERATIONS_SUBSET_SIZE, gates_weights.end());

  for (unsigned int i = 0; i < nr_operations; i++) {
    const unsigned int idx = operations_distribution(qc_rng());
    const auto [baseGate, isAdj, exactControls, minControls, allowExtraControls,
                isSwap] = gateSpecFromIndex(idx);

    unsigned int nr_targets = isSwap ? 2u : 1u;
    unsigned int nr_controls = 0u;
    // Regardless of what exactControls and minControls are, we cannot be
    // using more qubits than are available for use.
    if (exactControls >= 0) {
      nr_controls = std::min(static_cast<unsigned>(exactControls),
                             nr_qubits - nr_targets);
    } else if (minControls > 0) {
      nr_controls =
          std::min(static_cast<unsigned>(minControls), nr_qubits - nr_targets);
    }
    if (exactControls < 0 && allowExtraControls &&
        randomizer_options.probability_additionals_controls > 0.0) {
      while (nr_controls + nr_targets < nr_qubits &&
             random01() < randomizer_options.probability_additionals_controls) {
        nr_controls++;
      }
    }
    try {
      auto [targets, controls] = sample_distinct_targets_and_controls(
          nr_targets, nr_controls, nr_qubits);
      std::vector<double> angles = makeAngles(baseGate);
      insertGate(buildSetup, baseGate, targets, controls, angles, isAdj);

      if (baseGate == MX || baseGate == MY || baseGate == MZ) {
        // Measurement by default do not have controls so there is no need to
        // consider controls as possibly involved qubits.
        for (int qubit : targets) {
          depths[qubit]++;
        }
        continue;
      }
      std::vector<int> involvedQubits = targets;
      involvedQubits.insert(involvedQubits.end(), controls.begin(),
                            controls.end());
      unsigned int max_depth = 0;
      for (int qubit : involvedQubits) {
        max_depth = std::max(max_depth, depths[qubit]);
      }
      for (int qubit : involvedQubits) {
        depths[qubit] = max_depth + 1;
      }

    } catch (const std::runtime_error &error) {
      std::cerr << "\n"
                << error.what() << "\nGate: " << SUPPORTED_GATES[baseGate]
                << std::endl;
    }
  }
  for (unsigned int qubit = 0; qubit < nr_measurements; qubit++) {
    int qubit_idx = qubit % nr_qubits;
    std::vector targets{qubit_idx};
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
    depths[qubit_idx]++;
  }
  return {buildSetup.module, std::move(buildSetup.ctxOwner), nr_qubits,
          nr_gates, get_max_depth(depths)};
}

QuantumCircuit random_quantum_circuit_from_yaml_statistics(
    const fs::path &statistics_yaml_file_path,
    const RandomizerOptions &randomizer_options) {
  if (!fs::exists(statistics_yaml_file_path) ||
      !fs::is_regular_file(statistics_yaml_file_path)) {
    std::cerr << "File: " << statistics_yaml_file_path
              << " does not exist or is not a regular file." << std::endl;
  }
  //////////////////////////////////////////////////////////////////////////////
  // Load from YAML
  YAML::Node statistics = YAML::LoadFile(statistics_yaml_file_path.string());

  const YAML::Node &yaml_qubits_cholesky_params =
      statistics["qubits_cholesky_params"];
  const YAML::Node &yaml_gates_weights = statistics["gates_weights"];
  if (!yaml_qubits_cholesky_params || !yaml_gates_weights) {
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

  std::array<double, CHOLESKY_PARAMS_SIZE> cholesky_params = {};
  cholesky_params[MEAN_QUBITS_INDEX] =
      get_double(yaml_qubits_cholesky_params, "mean_qubits");
  cholesky_params[MEAN_GATES_INDEX] =
      get_double(yaml_qubits_cholesky_params, "mean_gates");
  cholesky_params[MEAN_OPERATIONS_INDEX] =
      get_double(yaml_qubits_cholesky_params, "mean_operations");
  cholesky_params[MEAN_MEASUREMENTS_INDEX] =
      get_double(yaml_qubits_cholesky_params, "mean_measurements");
  cholesky_params[QUBITS_L11_INDEX] =
      get_double(yaml_qubits_cholesky_params, "qubits_L11");
  cholesky_params[GATES_L21_INDEX] =
      get_double(yaml_qubits_cholesky_params, "gates_L21");
  cholesky_params[GATES_L22_INDEX] =
      get_double(yaml_qubits_cholesky_params, "gates_L22");
  cholesky_params[OPERATIONS_L21_INDEX] =
      get_double(yaml_qubits_cholesky_params, "operations_L21");
  cholesky_params[OPERATIONS_L22_INDEX] =
      get_double(yaml_qubits_cholesky_params, "operations_L22");
  cholesky_params[MEASUREMENTS_L21_INDEX] =
      get_double(yaml_qubits_cholesky_params, "measurements_L21");
  cholesky_params[MEASUREMENTS_L22_INDEX] =
      get_double(yaml_qubits_cholesky_params, "measurements_L22");

  std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights{};

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
      cholesky_params, gates_weights, randomizer_options);
}

} // namespace ai_pass_selector
