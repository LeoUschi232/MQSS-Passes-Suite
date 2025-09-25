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
//   Number of CX gates: 5623760
//   Number of CCX gates: 34803
//   Number of (3+)-controlled-X gates: 0
//   Number of Y gates: 5712
//   Number of controlled-Y gates: 16372
//   Number of Z gates: 6018
//   Number of controlled-Z gates: 44138
//   Number of H gates: 344383
//   Number of controlled-H gates: 16446
//   Number of S gates: 43873
//   Number of controlled-S gates: 0
//   Number of Sdg gates: 25324
//   Number of controlled-Sdg gates: 0
//   Number of T gates: 37419
//   Number of controlled-T gates: 0
//   Number of Tdg gates: 37417
//   Number of controlled-Tdg gates: 0
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
//   Number of gates with more than 1 target: 50497
//   Gates with more than 1 target: swap
//   Number of gates with more than 1 control: 34803
//   Gates with more than 1 control: x
//   Most targets in a gate: 2
//   Most controls in a gate: 2


std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_circuit_with_mqt_bench_statistics(int seed) {
  unsigned int nr_qubits = 130;
  unsigned int nr_instructions = 98338;

  for (int instr = 0; instr < nr_instructions; instr++) {
  }
  auto rebuildSetup =
      beginReconstruction("__nvqpp__mlirgen__FromTensor", nr_qubits);
  return {};
}

std::tuple<ModuleOp, std::unique_ptr<MLIRContext *>>
random_quantum_circuit(int seed, unsigned int nr_qubits, unsigned int nr_instructions) {
  throw std::runtime_error("Not implemented yet.");
}
} // namespace ai_pass_selector
