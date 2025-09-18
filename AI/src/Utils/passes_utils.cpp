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

} // namespace ai_pass_selector