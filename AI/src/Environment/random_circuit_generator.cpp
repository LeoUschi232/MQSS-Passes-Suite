#include "Environment/random_circuit_generator.hpp"

// Utils includes
#include "Utils/tensor_utils.hpp"

namespace ai_pass_selector {
// MQT Bench Dataset statistics
// Dataset name: MQTBench
//   Number of circuits: 1943
//   Minimum number of qubits: 0
//   Maximum number of qubits: 130
//   Total number of qubits: 120712
//   Minimum number of gates: 0
//   Maximum number of gates: 98338
//   Total number of gates: 10299578
//   Minimum depth: 0
//   Maximum depth: 98139
//   Total depth: 975666
//   Number of Mx gates: 0
//   Number of My gates: 0
//   Number of Mz gates: 120325
//   Number of X gates: 17022
//   Number of controlled-X gates: 5658563
//   Number of Y gates: 5712
//   Number of controlled-Y gates: 16372
//   Number of Z gates: 6018
//   Number of controlled-Z gates: 44138
//   Number of H gates: 344383
//   Number of controlled-H gates: 16446
//   Number of S gates: 69197
//   Number of controlled-S gates: 0
//   Number of T gates: 74836
//   Number of controlled-T gates: 0
//   Number of Rx gates: 26867
//   Number of controlled-Rx gates: 16275
//   Number of Ry gates: 192001
//   Number of controlled-Ry gates: 17014
//   Number of Rz gates: 73845
//   Number of controlled-Rz gates: 17321
//   Number of Swap gates: 33684
//   Number of controlled-Swap gates: 16813
//   Number of R1 gates: 878975
//   Number of controlled-R1 gates: 2225331
//   Number of U2 gates: 254545
//   Number of controlled-U2 gates: 0
//   Number of U3 gates: 141413
//   Number of controlled-U3 gates: 32482
//   Number of PhasedRx gates: 0
//   Number of controlled-PhasedRx gates: 0


std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit_by_nr_instructions(unsigned int nr_qubits,
                                          unsigned int nr_instructions) {

  for (int instr = 0; instr < nr_instructions; instr++) {
  }
  auto rebuildSetup =
      beginReconstruction("__nvqpp__mlirgen__FromTensor", nr_qubits);
  return {};
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
random_quantum_circuit_by_max_depth(unsigned int nr_qubits,
                                    unsigned int max_depth) {
  throw std::runtime_error("Not implemented yet.");
}
} // namespace ai_pass_selector