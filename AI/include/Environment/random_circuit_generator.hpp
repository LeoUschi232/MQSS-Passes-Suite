#ifndef RANDOM_CIRCUIT_GENERATOR_HPP
#define RANDOM_CIRCUIT_GENERATOR_HPP

#include "Support/mlir_utils.hpp"

namespace ai_pass_selector {
/**
 *
 * @param nr_qubits
 * @param nr_instructions
 * @return
 */
std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit_by_nr_instructions(unsigned int nr_qubits,
                                          unsigned int nr_instructions);
/**
 *
 * @param nr_qubits
 * @param max_instructions
 * @return
 */
std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit_by_max_instructions(unsigned int nr_qubits,
                                           unsigned int max_instructions);
/**
 *
 * @param nr_qubits
 * @param depth
 * @return
 */
std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit_by_depth(unsigned int nr_qubits, unsigned int depth);
/**
 *
 * @param nr_qubits
 * @param max_depth
 * @return
 */
std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit_by_max_depth(unsigned int nr_qubits, unsigned int max_depth);
} // namespace ai_pass_selector

#endif // MLIRPASSES_RANDOM_CIRCUIT_GENERATOR_H
