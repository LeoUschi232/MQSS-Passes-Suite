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

//                   X: 0.16716397936533683 %
//        controlled-X: 55.56972791501929 %
//                   Y: 0.05609450417899213 %
//        controlled-Y: 0.16078067619370784 %
//                   Z: 0.059099566902866704 %
//        controlled-Z: 0.43345574675286325 %
//                   H: 3.38200168639248 %
//        controlled-H: 0.16150739070863177 %
//                   S: 0.4308533231521222 %
//        controlled-S: 0.004998250612285699 %
//                  Sdg: 0.24869349156666612 %
//       controlled-Sdg: 0.004998250612285699 %
//                   T: 0.3674720328910551 %
//        controlled-T: 0.004998250612285699 %
//                  Tdg: 0.3674523919582193 %
//       controlled-Tdg: 0.004998250612285699 %
//                  Rx: 0.2638464712494716 %
//       controlled-Rx: 0.15982809095117245 %
//                  Ry: 1.8855393727014476 %
//       controlled-Ry: 0.1670854156339937 %
//                  Rz: 0.7251923426291446 %
//       controlled-Rz: 0.1701002988242862 %
//                Swap: 0.33079259082023305 %
//     controlled-Swap: 0.16511150188399767 %
//                  R1: 8.631944469665548 %
//       controlled-R1: 21.853788354191305 %
//                  U2: 2.499750624342008 %
//       controlled-U2: 0.004998250612285699 %
//                  U3: 1.3887416175531888 %
//       controlled-U3: 0.31898839018592834 %
//            PhasedRx: 0.004998250612285699 %
// controlled-PhasedRx: 0.004998250612285699 %

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
