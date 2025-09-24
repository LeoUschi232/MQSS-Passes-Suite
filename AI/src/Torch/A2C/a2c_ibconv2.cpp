#include "Torch/A2C/a2c_agents.hpp"

// Environment includes
#include "Environment/environment.hpp"

// Torch includes
#include "Torch/parallel_environments.hpp"

// Utils includes
#include "Utils/circuit_utils.hpp"
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <sstream>

namespace ai_pass_selector {
A2C_IBCONV2::A2C_IBCONV2(int circuit_size_class,
                         std::unordered_map<std::string, std::string> params)
    : BaseA2CAgent(circuit_size_class, std::move(params)) {
  unsigned int main_instr_repr_size =
      CIRCUIT_SIZE_CLASS_TO_MAIN_INSTR_REPR_SIZE.at(circuit_size_class);
  unsigned int lowdim_instr_repr_size =
      CIRCUIT_SIZE_CLASS_TO_LOWDIM_INSTR_REPR_SIZE.at(circuit_size_class);
}

std::string A2C_IBCONV2::agentName() const {
  std::string size_class_str = CIRCUIT_SIZE_CLASS_TO_NAME.at(this->size_class);
  std::ostringstream oss;
  oss << "a2c-" << size_class_str << "-ibconv2";
  return oss.str();
}
} // namespace ai_pass_selector
