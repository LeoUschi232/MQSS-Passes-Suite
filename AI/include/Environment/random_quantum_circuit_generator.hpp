#ifndef RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP
#define RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP

// Environment includes
#include "Environment/statistics_for_rqcg.hpp"

// Support includes
#include "Support/mlir_utils.hpp"

// Standard library includes
#include <filesystem>

namespace fs = std::filesystem;

namespace ai_pass_selector {
struct GateSpec {
  // X,Y,Z,H,S,T,RX,RY,RZ,SWAP,R1,U2,U3,PHASED_RX
  int baseGate;
  // Sdg/Tdg
  bool isAdj;
  // >=0 exact; -1 = not exact
  int exactControls;
  // minimum when not exact (0/1/2/3+)
  int minControls;
  // use probability_additional_controls
  bool allowExtraControls;
  // special: exactly 2 targets
  bool isSwap;
};

struct RandomizerOptions {
  // If exists, will seed the random generator using this seed
  std::optional<int> seed = std::nullopt;
  // If >=2, will make the circuit have at least this many qubits
  int min_nr_qubits = -1;
  // If >=2, will make the circuit have at most this many qubits
  // If max_nr_qubits < min_nr_qubits, max_nr_qubits overrides min_nr_qubits
  int max_nr_qubits = -1;
  // If >=2, use this exact number of qubits
  // and ignore min_nr_qubits and max_nr_qubits
  int exact_nr_qubits = -1;
  // If >=2, will make the circuit have at least this many non-measurement gates
  int min_nr_non_measurement_gates = -1;
  // If >=2, will make the circuit have at most this many non-measurement gates
  // If max_nr_gates < min_nr_gates, max_nr_gates overrides min_nr_gates
  int max_nr_non_measurement_gates = -1;
  // If >=2, use this exact number of non-measurement gates
  // and ignore min_nr_gates and max_nr_gates
  int exact_non_measurement_nr_gates = -1;
  // If >0.0, gates with zero-weights (not occurring in the statistics) will get
  // this times minimum weight of non-zero-weight gates weight.
  // Does not affect measurement gates.
  double weight_min_multiplier_for_unoccurring_gates = 0.0;
  // If >0.0, when a gate allows additional controls, this is the probability
  // for adding each additional control.
  double probability_additionals_controls = 0.0;
  //
  bool
};

/**
 *
 * @param idx
 * @return
 */
static GateSpec gateSpecFromIndex(unsigned int idx);

/**
 *
 * @param nr_targets
 * @param nr_controls
 * @param nr_qubits
 * @return
 */
std::pair<std::vector<int>, std::vector<int>> sampleDistinctTargetsAndControls(
    unsigned int nr_targets, unsigned int nr_controls, unsigned int nr_qubits);

/**
 *
 * @param baseGate
 * @return
 */
std::vector<double> makeAngles(int baseGate);

/**
 *
 * @param qubits_and_gates_distribution_params
 * @return
 */
std::pair<unsigned int, unsigned int> sample_nr_qubits_and_gates_from_cholesky(
    std::tuple<double, double, double, double, double>
        qubits_and_gates_distribution_params);

/**
 *
 * @param qubits_and_gates_distribution_params
 * @param gates_weights
 * @param randomizer_options
 * @return
 */
std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
random_quantum_circuit_from_embedded_statistics(
    const std::tuple<double, double, double, double, double>
        &qubits_and_gates_distribution_params,
    std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights,
    const RandomizerOptions &randomizer_options = {
        .weight_min_multiplier_for_unoccurring_gates = 0.1,
        .probability_additionals_controls = 0.01});

/**
 *
 * @param statistics_yaml_file_path
 * @param cap_nr_qubits
 * @param cap_nr_gates
 * @param give_small_probability_to_unoccurring_gates
 * @param probability_additionals_qubits
 * @return
 */
std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
random_quantum_circuit_from_yaml_statistics(
    const fs::path &statistics_yaml_file_path,
    const std::pair<int, int> &cap_nr_qubits = {-1, -1},
    const std::pair<int, int> &cap_nr_gates = {-1, -1},
    bool give_small_probability_to_unoccurring_gates = true,
    double probability_additionals_qubits = 0.01);
} // namespace ai_pass_selector

#endif // RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP
