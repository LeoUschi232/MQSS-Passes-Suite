#ifndef RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP
#define RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP

// Environment includes
#include "Environment/statistics_for_rqcg.hpp"

// Support includes
#include "Support/mlir_utils.hpp"

// Standard library includes
#include <filesystem>
#include <random>

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

/**
 *
 * @param idx
 * @return
 */
static GateSpec gateSpecFromIndex(unsigned int idx);

/**
 *
 * @param universe
 * @param required
 * @param extra_probability
 * @param forbidden
 * @return
 */
std::vector<int>
sampleDistinctQubits(unsigned universe, unsigned required,
                     double extra_probability,
                     const std::unordered_set<int> &forbidden = {});

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
 * @param give_small_probability_to_unoccurring_gates
 * @param probability_additionals_qubits
 * @return
 */
std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
random_quantum_circuit_from_embedded_statistics(
    const std::tuple<double, double, double, double, double>
        &qubits_and_gates_distribution_params,
    std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights,
    bool give_small_probability_to_unoccurring_gates = true,
    double probability_additionals_qubits = 0.01);

/**
 *
 * @param statistics_yaml_file_path
 * @param give_small_probability_to_unoccurring_gates
 * @param probability_additionals_qubits
 * @return
 */
std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
random_quantum_circuit_from_yaml_statistics(
    const fs::path &statistics_yaml_file_path,
    bool give_small_probability_to_unoccurring_gates = true,
    double probability_additionals_qubits = 0.01);
} // namespace ai_pass_selector

#endif // RANDOM_QUANTUM_CIRCUIT_GENERATOR_HPP
