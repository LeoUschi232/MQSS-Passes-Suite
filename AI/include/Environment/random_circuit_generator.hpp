#ifndef RANDOM_CIRCUIT_GENERATOR_HPP
#define RANDOM_CIRCUIT_GENERATOR_HPP

// Support includes
#include "Support/mlir_utils.hpp"

// Standard library includes
#include <random>

namespace ai_pass_selector {
/**
 *
 * @param rng
 * @return
 */
double rand01(std::mt19937 &rng);

/**
 *
 * @param rng
 * @return
 */
double randAngle(std::mt19937 &rng);

struct GateSpec {
  int baseGate;            // X,Y,Z,H,S,T,RX,RY,RZ,SWAP,R1,U2,U3,PHASED_RX
  bool isAdj;              // Sdg/Tdg
  int exactControls;       // >=0 exact; -1 = not exact
  int minControls;         // minimum when not exact (0/1/2/3+)
  bool allowExtraControls; // use probability_additional_controls
  bool isSwap;             // special: exactly 2 targets
};

/**
 *
 * @param seed
 * @param nr_qubits
 * @param nr_gates
 * @param gates_weights
 * @param measurement_weights
 * @param give_small_probability_to_unoccurring_gates
 * @param give_small_probability_to_unoccurring_measurements
 * @param probability_additional_targets
 * @param probability_additional_controls
 * @return
 */
std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>> random_quantum_circuit(
    int seed, unsigned int nr_qubits, unsigned int nr_gates,
    std::array<unsigned int, 34> gates_weights,
    std::array<unsigned int, 3> measurement_weights,
    bool give_small_probability_to_unoccurring_gates = true,
    bool give_small_probability_to_unoccurring_measurements = true,
    double probability_additional_targets = 0.01,
    double probability_additional_controls = 0.01);

/* MQT Bench Dataset statistics
Dataset name: MQTBench
   Number of circuits: 1943
   Minimum number of qubits: 2
   Maximum number of qubits: 130
   Total number of qubits: 120717
   Minimum number of gates: 4
   Maximum number of gates: 98338
   Total number of gates: 10299600
   Minimum depth: 3
   Maximum depth: 98139
   Total depth: 975677
   Number of Mx gates: 0
   Number of My gates: 0
   Number of Mz gates: 120330
   Number of X gates: 17022
   Number of CX gates: 5623760
   Number of CCX gates: 34803
   Number of (3+)-controlled-X gates: 0
   Number of Y gates: 5712
   Number of controlled-Y gates: 16372
   Number of Z gates: 6018
   Number of controlled-Z gates: 44138
   Number of H gates: 344388
   Number of controlled-H gates: 16446
   Number of S gates: 43873
   Number of controlled-S gates: 0
   Number of Sdg gates: 25324
   Number of controlled-Sdg gates: 0
   Number of T gates: 37419
   Number of controlled-T gates: 0
   Number of Tdg gates: 37417
   Number of controlled-Tdg gates: 0
   Number of Rx gates: 26867
   Number of controlled-Rx gates: 16275
   Number of Ry gates: 192001
   Number of controlled-Ry gates: 17014
   Number of Rz gates: 73845
   Number of controlled-Rz gates: 17321
   Number of Swap gates: 33686
   Number of controlled-Swap gates: 16813
   Number of R1 gates: 878975
   Number of controlled-R1 gates: 2225341
   Number of U2 gates: 254545
   Number of controlled-U2 gates: 0
   Number of U3 gates: 141413
   Number of controlled-U3 gates: 32482
   Number of PhasedRx gates: 0
   Number of controlled-PhasedRx gates: 0
   Number of gates with 2+ target: 50499
   Gates with 2+ target: swap
   Number of gates with 2+ control: 34803
   Gates with 2+ control: x
   Most targets in a gate: 2
   Most controls in a gate: 2
*/
constexpr std::array<unsigned int, 34> MQT_BENCH_GATES_WEIGHTS = {
    /* X */ 17022u,
    /* CX */ 5623760u,
    /* CCX */ 34803u,
    /* (3+)-controlled-X */ 0u,
    /* Y */ 5712u,
    /* controlled-Y */ 16372u,
    /* Z */ 6018u,
    /* controlled-Z */ 44138u,
    /* H */ 344388u,
    /* controlled-H */ 16446u,
    /* S */ 43873u,
    /* controlled-S */ 0u,
    /* Sdg */ 25324u,
    /* controlled-Sdg */ 0u,
    /* T */ 37419u,
    /* controlled-T */ 0u,
    /* Tdg */ 37417u,
    /* controlled-Tdg */ 0u,
    /* Rx */ 26867u,
    /* controlled-Rx */ 16275u,
    /* Ry */ 192001u,
    /* controlled-Ry */ 17014u,
    /* Rz */ 73845u,
    /* controlled-Rz */ 17321u,
    /* Swap */ 33686u,
    /* controlled-Swap */ 16813u,
    /* R1 */ 878975u,
    /* controlled-R1 */ 2225341u,
    /* U2 */ 254545u,
    /* controlled-U2 */ 0u,
    /* U3 */ 141413u,
    /* controlled-U3 */ 32482u,
    /* PhasedRx */ 0u,
    /* controlled-PhasedRx */ 0u};
constexpr std::array<unsigned int, 3> MQT_BENCH_MEASUREMENT_WEIGHTS = {
    /* Mx */ 0u,
    /* My */ 0u,
    /* Mz */ 120330};

/**
 *
 * @param seed
 * @return
 */
std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_circuit_with_mqt_bench_statistics(int seed);
} // namespace ai_pass_selector

#endif // RANDOM_CIRCUIT_GENERATOR_HPP
