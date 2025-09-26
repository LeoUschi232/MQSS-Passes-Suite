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

std::vector<int>
sampleDistinctQubits(unsigned universe, unsigned required,
                     double extra_probability,
                     const std::unordered_set<int> &forbidden) {
  if (required > universe - forbidden.size()) {
    required = universe - static_cast<unsigned>(forbidden.size());
  }
  std::uniform_int_distribution qubit_distribution(
      0, static_cast<int>(universe) - 1);
  std::unordered_set<int> chosen = forbidden;
  std::vector<int> out;
  out.reserve(required + 4);
  while (out.size() < required) {
    if (int qubit = qubit_distribution(rng); chosen.insert(qubit).second) {
      out.push_back(qubit);
    }
  }
  while (random01() < extra_probability && chosen.size() < universe) {
    if (int qubit = qubit_distribution(rng); chosen.insert(qubit).second) {
      out.push_back(qubit);
    }
  }
  return out;
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
    double z1 = ndist(qc_rng), z2 = ndist(qc_rng);
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
    bool give_small_probability_to_unoccurring_gates,
    double probability_additionals_qubits) {
  if (give_small_probability_to_unoccurring_gates) {
    // If a specific type of gate doesn't occur in the statistics, give it 1/10
    // of the weight of the least occurring gate but at least a weight of 1.
    unsigned int minimum_weight = std::numeric_limits<unsigned int>::max();
    for (const auto &weight : gates_weights) {
      if (weight > 0 && weight < minimum_weight) {
        minimum_weight = weight;
      }
    }
    minimum_weight = std::max(1u, minimum_weight / 10u);
    for (unsigned int i = 0; i < gates_weights.size(); i++) {
      if (gates_weights[i] <= 0) {
        gates_weights[i] = minimum_weight;
      }
    }
  }
  auto [nr_qubits, nr_gates] = sample_nr_qubits_and_gates_from_cholesky(
      qubits_and_gates_distribution_params);

  auto buildSetup = beginReconstruction("__nvqpp__mlirgen__Random", nr_qubits);
  for (unsigned int i = 0; i < nr_gates - nr_qubits; i++) {
    // Sample a gate according to operations-only subset of gates_weights.
    std::discrete_distribution<size_t> operations_distribution(
        gates_weights.begin(), gates_weights.begin() + OPERATIONS_SUBSET_SIZE);
    const unsigned int idx =
        static_cast<unsigned int>(operations_distribution(qc_rng));
    const auto [baseGate, isAdj, exactControls, minControls, allowExtraControls,
                isSwap] = gateSpecFromIndex(idx);

    std::vector<int> targets =
        isSwap ? sampleDistinctQubits(nr_qubits, /*required=*/2, /*extra=*/0.0)
               : sampleDistinctQubits(nr_qubits, /*required=*/1,
                                      /*extra=*/probability_additionals_qubits);
    std::unordered_set forbid(targets.begin(), targets.end());
    std::vector<int> controls;
    if (exactControls >= 0) {
      controls =
          sampleDistinctQubits(nr_qubits, static_cast<unsigned>(exactControls),
                               /*extra=*/0.0, forbid);
    } else if (minControls > 0) {
      controls = sampleDistinctQubits(
          nr_qubits, static_cast<unsigned>(minControls),
          allowExtraControls ? probability_additionals_qubits : 0.0, forbid);
    }
    // Angles if any and emit the operation.
    std::vector<double> angles = makeAngles(baseGate);
    insertGate(buildSetup, baseGate, isAdj, controls, targets, angles);
  }
  return {buildSetup.module, std::move(buildSetup.ctxOwner)};
}

} // namespace ai_pass_selector
