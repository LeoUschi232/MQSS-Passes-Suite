#include "Environment/random_circuit_generator.hpp"


namespace ai_pass_selector {

std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit_by_nr_instructions(unsigned int nr_qubits,
                                          unsigned int nr_instructions) {




}

std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit_by_max_instructions(unsigned int nr_qubits,
                                           unsigned int max_instructions) {
  throw std::runtime_error("Not implemented yet.");
}
std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit_by_depth(unsigned int nr_qubits, unsigned int depth) {
  throw std::runtime_error("Not implemented yet.");
}
std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit_by_max_depth(unsigned int nr_qubits, unsigned int max_depth) {
  throw std::runtime_error("Not implemented yet.");
}
} // namespace ai_pass_selector