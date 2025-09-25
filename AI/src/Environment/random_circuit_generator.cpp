#include "Environment/random_circuit_generator.hpp"

// Utils includes
#include "Interfaces/Constants.hpp"
#include "Utils/tensor_utils.hpp"

// Standard library includes
#include <random>
#include <unordered_set>

namespace ai_pass_selector {

double rand01(std::mt19937 &rng) {
  thread_local std::uniform_real_distribution distribution(0.0, 1.0);
  return distribution(rng);
}
double randAngle(std::mt19937 &rng) {
  thread_local std::uniform_real_distribution distribution(0.0, 2.0 * PI);
  return distribution(rng);
}

// 34 weighted “kinds” -> GateSpec
static GateSpec gateSpecFromIndex(unsigned idx) {
  switch (idx) {
  case 0:
    return {X, false, 0, 0, false, false}; // X
  case 1:
    return {X, false, 1, 0, false, false}; // CX
  case 2:
    return {X, false, 2, 0, false, false}; // CCX
  case 3:
    return {X, false, -1, 3, true, false}; // (3+)-controlled-X
  case 4:
    return {Y, false, 0, 0, false, false}; // Y
  case 5:
    return {Y, false, -1, 1, true, false}; // controlled-Y
  case 6:
    return {Z, false, 0, 0, false, false}; // Z
  case 7:
    return {Z, false, -1, 1, true, false}; // controlled-Z
  case 8:
    return {H, false, 0, 0, false, false}; // H
  case 9:
    return {H, false, -1, 1, true, false}; // controlled-H
  case 10:
    return {S, false, 0, 0, false, false}; // S
  case 11:
    return {S, false, -1, 1, true, false}; // controlled-S
  case 12:
    return {S, true, 0, 0, false, false}; // Sdg
  case 13:
    return {S, true, -1, 1, true, false}; // controlled-Sdg
  case 14:
    return {T, false, 0, 0, false, false}; // T
  case 15:
    return {T, false, -1, 1, true, false}; // controlled-T
  case 16:
    return {T, true, 0, 0, false, false}; // Tdg
  case 17:
    return {T, true, -1, 1, true, false}; // controlled-Tdg
  case 18:
    return {RX, false, 0, 0, false, false}; // Rx
  case 19:
    return {RX, false, -1, 1, true, false}; // controlled-Rx
  case 20:
    return {RY, false, 0, 0, false, false}; // Ry
  case 21:
    return {RY, false, -1, 1, true, false}; // controlled-Ry
  case 22:
    return {RZ, false, 0, 0, false, false}; // Rz
  case 23:
    return {RZ, false, -1, 1, true, false}; // controlled-Rz
  case 24:
    return {SWAP, false, 0, 0, false, true}; // Swap (2 targets)
  case 25:
    return {SWAP, false, -1, 1, true, true}; // controlled-Swap (>=1 control)
  case 26:
    return {R1, false, 0, 0, false, false}; // R1
  case 27:
    return {R1, false, -1, 1, true, false}; // controlled-R1
  case 28:
    return {U2, false, 0, 0, false, false}; // U2
  case 29:
    return {U2, false, -1, 1, true, false}; // controlled-U2
  case 30:
    return {U3, false, 0, 0, false, false}; // U3
  case 31:
    return {U3, false, -1, 1, true, false}; // controlled-U3
  case 32:
    return {PHASED_RX, false, 0, 0, false, false}; // PhasedRx
  case 33:
    return {PHASED_RX, false, -1, 1, true, false}; // controlled-PhasedRx
  default:
    llvm::report_fatal_error("gateSpecFromIndex: out of range");
  }
}

static std::vector<int>
sampleDistinctQubits(std::mt19937 &rng, unsigned universe, unsigned required,
                     double extra_probability,
                     const std::unordered_set<int> &forbidden = {}) {
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
  while (rand01(rng) < extra_probability && chosen.size() < universe) {
    if (int qubit = qubit_distribution(rng); chosen.insert(qubit).second) {
      out.push_back(qubit);
    }
  }
  return out;
}

static std::vector<double> makeAngles(int baseGate, std::mt19937 &rng) {
  switch (baseGate) {
  case RX:
  case RY:
  case RZ:
  case R1:
    return {randAngle(rng)};
  case U2:
    return {randAngle(rng), randAngle(rng)};
  case U3:
    return {randAngle(rng), randAngle(rng), randAngle(rng)};
  case PHASED_RX:
    return {randAngle(rng), randAngle(rng)};
  default:
    return {};
  }
}

std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit(int seed, unsigned int nr_qubits, unsigned int nr_gates,
                       std::array<unsigned int, 34> gates_weights,
                       std::array<unsigned int, 3> measurement_weights,
                       bool give_small_probability_to_unoccurring_gates,
                       bool give_small_probability_to_unoccurring_measurements,
                       double probability_additional_targets,
                       double probability_additional_controls) {
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
  if (give_small_probability_to_unoccurring_measurements) {
    unsigned int minimum_weight = std::numeric_limits<unsigned int>::max();
    for (const auto &weight : measurement_weights) {
      if (weight > 0 && weight < minimum_weight) {
        minimum_weight = weight;
      }
    }
    minimum_weight = std::max(1u, minimum_weight / 100u);
    for (unsigned int i = 0; i < measurement_weights.size(); i++) {
      if (measurement_weights[i] <= 0) {
        measurement_weights[i] = minimum_weight;
      }
    }
  }

  std::mt19937 rng(static_cast<unsigned>(seed));
  auto setup = beginReconstruction("__nvqpp__mlirgen__Random", nr_qubits);
  std::discrete_distribution<size_t> gate_distribution(gates_weights.begin(),
                                                       gates_weights.end());
  std::discrete_distribution<size_t> measure_distribution(
      measurement_weights.begin(), measurement_weights.end());
  const unsigned opsToEmit = nr_gates > nr_qubits ? nr_gates - nr_qubits : 0;

  // Emit operating gates
  for (unsigned i = 0; i < opsToEmit; ++i) {
    unsigned idx = static_cast<unsigned>(gate_distribution(rng));
    auto [baseGate, isAdj, exactControls, minControls, allowExtraCtrls,
          isSwap] = gateSpecFromIndex(idx);

    std::vector<int> targets;
    if (isSwap) {
      targets = sampleDistinctQubits(rng, nr_qubits, /*required=*/2,
                                     /*extra_probability=*/0.0);
    } else {
      targets = sampleDistinctQubits(rng, nr_qubits, /*required=*/1,
                                     probability_additional_targets);
    }
    std::unordered_set forbid(targets.begin(), targets.end());
    std::vector<int> controls;
    if (exactControls >= 0) {
      const unsigned need = static_cast<unsigned>(exactControls);
      controls = sampleDistinctQubits(rng, nr_qubits, need,
                                      /*extra_probability=*/0.0, forbid);
    } else if (minControls > 0) {
      controls = sampleDistinctQubits(
          rng, nr_qubits, static_cast<unsigned>(minControls),
          allowExtraCtrls ? probability_additional_controls : 0.0, forbid);
    }

    std::vector<double> angles = makeAngles(baseGate, rng);
    insertGate(setup, baseGate, isAdj, controls, targets, angles);
  }

  // Measure every qubit once
  for (unsigned q = 0; q < nr_qubits; ++q) {
    unsigned m = static_cast<unsigned>(measure_distribution(rng));
    int gateIndex = m == 0 ? MX : m == 1 ? MY : MZ;
    std::vector targets{static_cast<int>(q)};
    std::vector<int> controls;
    std::vector<double> angles;
    insertGate(setup, gateIndex, /*isAdj=*/false, controls, targets, angles);
  }

  // Convert ctx unique_ptr<MLIRContext> -> unique_ptr<MLIRContext*>
  MLIRContext *raw = setup.ctxOwner.release();
  auto ctxPtrPtr = std::make_unique<MLIRContext *>(raw);
  return {setup.module, std::move(ctxPtrPtr)};
}

std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_circuit_with_mqt_bench_statistics(int seed) {
  return random_quantum_circuit(
      seed, /*nr_qubits=*/130, /*nr_gates=*/98338,
      /*gates_weights=*/MQT_BENCH_GATES_WEIGHTS,
      /*measurement_weights=*/MQT_BENCH_MEASUREMENT_WEIGHTS);
}

} // namespace ai_pass_selector
