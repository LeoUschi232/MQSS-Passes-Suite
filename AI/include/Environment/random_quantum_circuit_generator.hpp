#ifndef RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP
#define RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP

// Environment includes
#include "Environment/quantum_circuit.hpp"
#include "Environment/statistics_for_rqcg.hpp"

// Support includes
#include "Support/mlir_utils.hpp"

// Standard library includes
#include <filesystem>

namespace fs = std::filesystem;

namespace ai_pass_selector {
enum class GateSymbol : int;
struct GateSpec {
  // X,Y,Z,H,S,T,RX,RY,RZ,SWAP,R1,U2,U3,PHASED_RX
  GateSymbol baseGate;
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
  // IF >=2 the nr of this object will be at least this mean value
  mutable int min_nr_qubits = -1;
  mutable int min_nr_gates = -1;
  mutable int min_nr_operations = -1;
  // If >=2 the nr of this object will be at most this mean value
  // max overrides min for all values
  mutable int max_nr_qubits = -1;
  mutable int max_nr_gates = -1;
  mutable int max_nr_operations = -1;
  // If >=2 the nr of this object will be exactly this value
  // exact overrides min and max for all values
  // All values must be >=2 and exact_nr_operations <= exact_nr_measurements
  // for these values to take effect.
  mutable int exact_nr_qubits = -1;
  mutable int exact_nr_gates = -1;
  mutable int exact_nr_operations = -1;
  // Whether to sample measurement gates along with operation gates
  mutable bool allow_measurements_as_gates = false;
  // If >0.0 for every gate with zero weight, its weight will be set to this
  // multiplier times the smallest non-zero weight in the gates_weights array
  mutable double weight_min_multiplier_for_unoccurring_gates = 0.0;
  // If >0.0 will add additional controls to a controlled gate with this
  // probability
  mutable double probability_additionals_controls = 0.0;
  // If >0.0 will make the circuit have exactly max_qubits with this probability
  // to give a probability of triggering all possible qubits.
  mutable double probability_max_qubits = 0.0;
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
std::pair<std::vector<int>, std::vector<int>>
sample_distinct_targets_and_controls(unsigned int nr_targets,
                                     unsigned int nr_controls,
                                     unsigned int nr_qubits);

/**
 *
 * @param baseGate
 * @return
 */
std::vector<float> makeAngles(GateSymbol baseGate);

/**
 *
 * @param cholesky_params
 * @return [nr_qubits, nr_gates, nr_operations, nr_measurements]
 */
std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>
sample_nr_qubits_gates_operations_measurements(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params);

/**
 *
 * @param cholesky_params
 * @param randomizer_options
 * @return [nr_qubits, nr_gates, nr_operations, nr_measurements]
 */
std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>
get_nr_qubits_gates_operations_measurements(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
    const RandomizerOptions &randomizer_options);

/**
 *
 * @param multiplier
 * @param subset_size
 * @param gates_weights
 */
void adjust_gates_weights(
    double multiplier, unsigned int subset_size,
    std::array<unsigned int, GATES_WEIGHTS_SIZE> &gates_weights);

/**
 *
 * @param cholesky_params
 * @param gates_weights
 * @param randomizer_options
 * @return
 */
QuantumCircuit random_quantum_circuit_from_embedded_statistics(
    const std::array<double, CHOLESKY_PARAMS_SIZE> &cholesky_params,
    std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights,
    const RandomizerOptions &randomizer_options = {
        .weight_min_multiplier_for_unoccurring_gates = 0.1,
        .probability_additionals_controls = 0.01});

/**
 *
 * @param statistics_yaml_file_path
 * @param randomizer_options
 * @return
 */
QuantumCircuit random_quantum_circuit_from_yaml_statistics(
    const fs::path &statistics_yaml_file_path,
    const RandomizerOptions &randomizer_options = {
        .weight_min_multiplier_for_unoccurring_gates = 0.1,
        .probability_additionals_controls = 0.01});
} // namespace ai_pass_selector

#endif // RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP
