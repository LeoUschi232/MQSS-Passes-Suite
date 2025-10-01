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
  mutable int min_nr_qubits = -1;
  // If >=2, will make the circuit have at most this many qubits
  // If max_nr_qubits < min_nr_qubits, max_nr_qubits overrides min_nr_qubits
  mutable int max_nr_qubits = -1;
  // If >=2, will make the circuit have at least this many non-measurement gates
  mutable int min_nr_non_measurement_gates = -1;
  // If >=2, will make the circuit have at most this many non-measurement gates
  // If max_nr_gates < min_nr_gates, max_nr_gates overrides min_nr_gates
  mutable int max_nr_non_measurement_gates = -1;
  // If >=2, use this exact number of qubits and ignore min_nr_qubits and
  // max_nr_qubits.
  // If >=2, use this exact number of non-measurement gates
  // and ignore min_nr_gates and max_nr_gates
  mutable std::pair<int, int> exact_nr_qubits_and_non_measurement_gates = {-1,
                                                                           -1};
  // Whether measurment gates can be sampled as regular gates
  mutable bool allow_measurements_as_gates = false;
  // If >0.0, gates with zero-weights (not occurring in the statistics) will get
  // this times minimum weight of non-zero-weight gates weight.
  // Affects measurement gates only if allow_measurements_as_gates=true
  mutable double weight_min_multiplier_for_unoccurring_gates = 0.0;
  // If >0.0, when a gate allows additional controls, this is the probability
  // for adding each additional control.
  mutable double probability_additionals_controls = 0.0;
  // Whether to measure all qubits at the end of the circuit
  // WARNING: Increases nr of gates by nr of qubits
  mutable bool measure_all_at_the_end = false;
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
std::pair<std::vector<int>, std::vector<int>> sample_distinct_targets_and_controls(
    unsigned int nr_targets, unsigned int nr_controls, unsigned int nr_qubits);

/**
 *
 * @param baseGate
 * @return
 */
std::vector<double> makeAngles(int baseGate);

/**
 *
 * @param cholesky_params
 * @return
 */
std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>
sample_nr_qubits_gates_operations_measurements(
    const std::array<unsigned int, CHOLESKY_PARAMS_SIZE> &cholesky_params);

/**
 *
 * @param cholesky_params
 * @param gates_weights
 * @param randomizer_options
 * @return
 */
std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
random_quantum_circuit_from_embedded_statistics(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
    std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights,
    const RandomizerOptions &randomizer_options = {
        .weight_min_multiplier_for_unoccurring_gates = 0.1,
        .probability_additionals_controls = 0.01,
        .measure_all_at_the_end = true});

/**
 *
 * @param statistics_yaml_file_path
 * @param randomizer_options
 * @return
 */
std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
random_quantum_circuit_from_yaml_statistics(
    const fs::path &statistics_yaml_file_path,
    const RandomizerOptions &randomizer_options = {
        .weight_min_multiplier_for_unoccurring_gates = 0.1,
        .probability_additionals_controls = 0.01,
        .measure_all_at_the_end = true});
} // namespace ai_pass_selector

#endif // RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP
