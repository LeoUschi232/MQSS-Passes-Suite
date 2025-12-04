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
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace fs = std::filesystem;
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {

static GateSpec gateSpecFromIndex(unsigned int idx) {
  if (idx >= GATES_WEIGHTS_SIZE) {
    llvm::report_fatal_error("gateSpecFromIndex: out of range");
  }
  switch (static_cast<GateWeightIndex>(idx)) {
  case GateWeightIndex::X:
    return {GateSymbol::X, false, 0, 0, false, false};
  case GateWeightIndex::CX:
    return {GateSymbol::X, false, 1, 0, false, false};
  case GateWeightIndex::CCX:
    return {GateSymbol::X, false, 2, 0, false, false};
  case GateWeightIndex::C3PlusX:
    return {GateSymbol::X, false, -1, 3, true, false};
  case GateWeightIndex::Y:
    return {GateSymbol::Y, false, 0, 0, false, false};
  case GateWeightIndex::ControlledY:
    return {GateSymbol::Y, false, -1, 1, true, false};
  case GateWeightIndex::Z:
    return {GateSymbol::Z, false, 0, 0, false, false};
  case GateWeightIndex::ControlledZ:
    return {GateSymbol::Z, false, -1, 1, true, false};
  case GateWeightIndex::H:
    return {GateSymbol::H, false, 0, 0, false, false};
  case GateWeightIndex::ControlledH:
    return {GateSymbol::H, false, -1, 1, true, false};
  case GateWeightIndex::S:
    return {GateSymbol::S, false, 0, 0, false, false};
  case GateWeightIndex::ControlledS:
    return {GateSymbol::S, false, -1, 1, true, false};
  case GateWeightIndex::SDG:
    return {GateSymbol::S, true, 0, 0, false, false};
  case GateWeightIndex::ControlledSDG:
    return {GateSymbol::S, true, -1, 1, true, false};
  case GateWeightIndex::T:
    return {GateSymbol::T, false, 0, 0, false, false};
  case GateWeightIndex::ControlledT:
    return {GateSymbol::T, false, -1, 1, true, false};
  case GateWeightIndex::TDG:
    return {GateSymbol::T, true, 0, 0, false, false};
  case GateWeightIndex::ControlledTDG:
    return {GateSymbol::T, true, -1, 1, true, false};
  case GateWeightIndex::RX:
    return {GateSymbol::RX, false, 0, 0, false, false};
  case GateWeightIndex::ControlledRX:
    return {GateSymbol::RX, false, -1, 1, true, false};
  case GateWeightIndex::RY:
    return {GateSymbol::RY, false, 0, 0, false, false};
  case GateWeightIndex::ControlledRY:
    return {GateSymbol::RY, false, -1, 1, true, false};
  case GateWeightIndex::RZ:
    return {GateSymbol::RZ, false, 0, 0, false, false};
  case GateWeightIndex::ControlledRZ:
    return {GateSymbol::RZ, false, -1, 1, true, false};
  case GateWeightIndex::SWAP:
    return {GateSymbol::SWAP, false, 0, 0, false, true};
  case GateWeightIndex::ControlledSWAP:
    return {GateSymbol::SWAP, false, -1, 1, true, true};
  case GateWeightIndex::R1:
    return {GateSymbol::R1, false, 0, 0, false, false};
  case GateWeightIndex::ControlledR1:
    return {GateSymbol::R1, false, -1, 1, true, false};
  case GateWeightIndex::U2:
    return {GateSymbol::U2, false, 0, 0, false, false};
  case GateWeightIndex::ControlledU2:
    return {GateSymbol::U2, false, -1, 1, true, false};
  case GateWeightIndex::U3:
    return {GateSymbol::U3, false, 0, 0, false, false};
  case GateWeightIndex::ControlledU3:
    return {GateSymbol::U3, false, -1, 1, true, false};
  case GateWeightIndex::PhasedRX:
    return {GateSymbol::PHASED_RX, false, 0, 0, false, false};
  case GateWeightIndex::ControlledPhasedRX:
    return {GateSymbol::PHASED_RX, false, -1, 1, true, false};
  case GateWeightIndex::MX:
    return {GateSymbol::MX, false, 0, 0, false, false};
  case GateWeightIndex::MY:
    return {GateSymbol::MY, false, 0, 0, false, false};
  case GateWeightIndex::MZ:
    return {GateSymbol::MZ, false, 0, 0, false, false};
  default:
    throw std::logic_error("gateSpecFromIndex: unknown index " +
                           std::to_string(idx));
  }
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

std::vector<float> makeAngles(GateSymbol baseGate) {
  switch (baseGate) {
  case GateSymbol::RX:
  case GateSymbol::RY:
  case GateSymbol::RZ:
  case GateSymbol::R1:
    return {randomAngle()};
  case GateSymbol::U2:
  case GateSymbol::PHASED_RX:
    return {randomAngle(), randomAngle()};
  case GateSymbol::U3:
    return {randomAngle(), randomAngle(), randomAngle()};
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
    if (randomizer_options.probability_max_qubits > 0.0 &&
        random01() < randomizer_options.probability_max_qubits &&
        randomizer_options.max_nr_qubits >=
            static_cast<int>(GLOBAL_MIN_NR_QUBITS)) {
      nr_qubits = randomizer_options.max_nr_qubits;
    }
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
    std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights) {
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

QuantumCircuit random_quantum_circuit_from_statistics_arrays(
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
      std::vector<float> angles = makeAngles(baseGate);
      insertGate(buildSetup, baseGate, targets, controls, angles, isAdj);

      if (baseGate == GateSymbol::MX || baseGate == GateSymbol::MY ||
          baseGate == GateSymbol::MZ) {
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
                << error_no_stacktrace(error)
                << "\nGate: " << SUPPORTED_GATES[static_cast<int>(baseGate)]
                << std::endl;
    }
  }
  for (unsigned int qubit = 0; qubit < nr_measurements; qubit++) {
    int qubit_idx = qubit % nr_qubits;
    std::vector targets{qubit_idx};
    if (const unsigned int idx =
            measurements_distribution(qc_rng()) + OPERATIONS_SUBSET_SIZE;
        static_cast<GateWeightIndex>(idx) == GateWeightIndex::MX) {
      insertGate(buildSetup, GateSymbol::MX, targets);
    } else if (static_cast<GateWeightIndex>(idx) == GateWeightIndex::MY) {
      insertGate(buildSetup, GateSymbol::MY, targets);
    } else if (static_cast<GateWeightIndex>(idx) == GateWeightIndex::MZ) {
      insertGate(buildSetup, GateSymbol::MZ, targets);
    } else {
      throw std::runtime_error("Unknown measurement index " +
                               std::to_string(idx));
    }
    depths[qubit_idx]++;
  }
  return {buildSetup.module, std::move(buildSetup.ctxOwner), nr_qubits,
          nr_gates, get_max_depth(depths)};
}

QuantumCircuit random_quantum_circuit_from_statistics_yaml_file(
    const fs::path &statistics_yaml_file,
    const RandomizerOptions &randomizer_options) {
  // Unsafe value() access on purpose.
  // The code is supposed to throw an error if the yaml file does not exist.
  auto [cholesky_params, gates_weights] =
      get_dataset_statistics_from_yaml_file(statistics_yaml_file).value();
  return random_quantum_circuit_from_statistics_arrays(
      cholesky_params, gates_weights, randomizer_options);
}

} // namespace ai_pass_selector
