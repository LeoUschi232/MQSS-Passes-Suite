#ifndef RANDOM_CIRCUIT_GENERATOR_HPP
#define RANDOM_CIRCUIT_GENERATOR_HPP

#include "Support/mlir_utils.hpp"

namespace ai_pass_selector {
constexpr std::array<unsigned int, 34> MQT_GATES_WEIGHTS = {
    /* X                   */ 17022u,
    /* CX                  */ 5623760u,
    /* CCX                 */ 34803u,
    /* (3+)-controlled-X   */ 0u,
    /* Y                   */ 5712u,
    /* controlled-Y        */ 16372u,
    /* Z                   */ 6018u,
    /* controlled-Z        */ 44138u,
    /* H                   */ 344383u,
    /* controlled-H        */ 16446u,
    /* S                   */ 43873u,
    /* controlled-S        */ 0u,
    /* Sdg                 */ 25324u,
    /* controlled-Sdg      */ 0u,
    /* T                   */ 37419u,
    /* controlled-T        */ 0u,
    /* Tdg                 */ 37417u,
    /* controlled-Tdg      */ 0u,
    /* Rx                  */ 26867u,
    /* controlled-Rx       */ 16275u,
    /* Ry                  */ 192001u,
    /* controlled-Ry       */ 17014u,
    /* Rz                  */ 73845u,
    /* controlled-Rz       */ 17321u,
    /* Swap                */ 33684u,
    /* controlled-Swap     */ 16813u,
    /* R1                  */ 878975u,
    /* controlled-R1       */ 2225331u,
    /* U2                  */ 254545u,
    /* controlled-U2       */ 0u,
    /* U3                  */ 141413u,
    /* controlled-U3       */ 32482u,
    /* PhasedRx            */ 0u,
    /* controlled-PhasedRx */ 0u};

/**
 *
 * @param seed
 * @return
 */
std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_circuit_with_mqt_bench_statistics(int seed);

std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit(int seed, unsigned int nr_qubits,
                       unsigned int max_instructions);
} // namespace ai_pass_selector

#endif // MLIRPASSES_RANDOM_CIRCUIT_GENERATOR_H
