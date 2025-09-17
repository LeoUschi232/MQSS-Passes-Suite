#include <iostream>

#include "Environment/quantum_circuit_tensor.hpp"
#include "Utils/passes_utils.hpp"


namespace ai_pass_selector {


std::pair<std::string, std::unique_ptr<mlir::Pass> >
getPassNameAndPointer(unsigned int index) {
  if (index >= NR_PASSES) {
    std::cerr << "In getPassByIndex: " << index << std::endl;
    return {"", nullptr};
  }
  std::unique_ptr<mlir::Pass> pass = PASS_FUNCTIONS[index]();
  return {std::string(pass.get()->getArgument()), std::move(pass)};
}

unsigned int getNrOfInputValuesForInstructionBased(
    unsigned int max_qubits, unsigned int max_instructions) {
  return max_instructions * (max_qubits + NR_GATES + MAX_GATE_PARAMS);
}

unsigned int getNrOfInputValuesForDepthBased(
    unsigned int max_qubits, unsigned int max_depth) {
  return max_depth * max_qubits
         * (NR_GATES + MAX_GATE_PARAMS + QUBIT_ROLE + max_qubits);
}
} // namespace ai_pass_selector