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

//                   X: 0.16722248675811477 %
//       controlled-X: 55.58917731978957 %
//                  Y: 0.0561141372554548 %
//       controlled-Y: 0.1608369494303757 %
//                  Z: 0.05912025175128273 %
//       controlled-Z: 0.4336074562642268 %
//                  H: 3.3831853869827184 %
//       controlled-H: 0.16156391829537983 %
//                  S: 0.43100412181522557 %
//       controlled-S: 0.0 %
//                 Sdg: 0.2487805342887145 %
//      controlled-Sdg: 0.0 %
//                  T: 0.36760064810256704 %
//       controlled-T: 0.0 %
//                 Tdg: 0.3675810002954048 %
//      controlled-Tdg: 0.0 %
//                 Rx: 0.26393881751440895 %
//      controlled-Rx: 0.1598840307830054 %
//                 Ry: 1.8861993114818936 %
//      controlled-Ry: 0.16714389552946568 %
//                 Rz: 0.725446159949065 %
//      controlled-Rz: 0.17015983392887474 %
//               Swap: 0.3309083682270202 %
//    controlled-Swap: 0.16516929090965712 %
//                 R1: 8.634965650229933 %
//      controlled-R1: 21.861437180115278 %
//                 U2: 2.500625537060529 %
//      controlled-U2: 0.0 %
//                 U3: 1.3892276771193328 %
//      controlled-U3: 0.3191000361224935 %
//           PhasedRx: 0.0 %
//controlled-PhasedRx: 0.0 %

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
