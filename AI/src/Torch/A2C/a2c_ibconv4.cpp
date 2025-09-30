#include "Torch/A2C/a2c_agents.hpp"

// Environment includes
#include "Environment/environment.hpp"

// Torch includes
#include "Torch/parallel_environments.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <sstream>
#include <stdexcept>

namespace ai_pass_selector {
A2C_IBCONV4::A2C_IBCONV4(int circuit_size_class,
                         std::unordered_map<std::string, std::string> params)
    : BaseA2CAgent(circuit_size_class, std::move(params)) {
  unsigned int kernel_size = max_qubits + NR_GATES + MAX_GATE_PARAMS;
  unsigned int stride = kernel_size;

  throw std::runtime_error("A2C_IB_CONV_LSD not implemented yet.");
}

std::string A2C_IBCONV4::agentName() const {
  std::string size_class_str = CIRCUIT_SIZE_CLASS_TO_NAME.at(this->size_class);
  std::ostringstream oss;
  oss << "a2c-" << size_class_str << "-ibconvlsd";
  return oss.str();
}
} // namespace ai_pass_selector
