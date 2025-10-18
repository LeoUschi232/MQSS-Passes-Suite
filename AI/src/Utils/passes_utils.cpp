#include "Utils/passes_utils.hpp"

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

namespace ai_pass_selector {

std::pair<std::string, std::unique_ptr<Pass>>
getPassNameAndPointer(unsigned int index) {
  if (index >= NR_PASSES) {
    std::cerr << "In getPassByIndex: index " << index << ">=" << NR_PASSES
              << " nr passes." << std::endl;
    return {"", nullptr};
  }
  std::unique_ptr<Pass> pass = PASS_FUNCTIONS[index]();
  // Just to be safe extract the name before moving the unique_ptr.
  auto name = std::string(pass.get()->getArgument());
  return {name, std::move(pass)};
}

} // namespace ai_pass_selector