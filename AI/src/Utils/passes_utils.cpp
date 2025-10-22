#include "Utils/passes_utils.hpp"

namespace ai_pass_selector {

std::pair<std::string, std::unique_ptr<Pass>>
getPassNameAndPointer(unsigned int index) {
  if (index >= NR_PASSES) {
    std::cerr << "In getPassByIndex: index " << index << ">=" << NR_PASSES
              << " nr passes." << std::endl;
    return {"", nullptr};
  }
  std::unique_ptr<Pass> pass_ptr = PASS_FUNCTIONS[index]();
  // Just to be safe extract the name before moving the unique_ptr.
  auto name = std::string(pass_ptr.get()->getArgument());
  return {name, std::move(pass_ptr)};
}

} // namespace ai_pass_selector