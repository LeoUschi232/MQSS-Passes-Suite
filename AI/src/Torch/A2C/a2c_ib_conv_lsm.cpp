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
A2C_IB_CONV_LSM::A2C_IB_CONV_LSM(
    int circuit_size_class, std::unordered_map<std::string, std::string> params)
: BaseA2CAgent(circuit_size_class, std::move(params)) {
  unsigned int kernel_size = max_qubits + NR_GATES + MAX_GATE_PARAMS;
  unsigned int conv_output_size = max_instructions;
  unsigned int nr_input_values = conv_output_size * kernel_size;
  auto critic = torch::nn::Sequential(
  torch::nn::ConvolutionalLayer(1, 3, kernel_size),



}

std::string A2C_IB_CONV_LSM::agentName() const {
  std::string size_class_str = CIRCUIT_SIZE_CLASS_TO_NAME.at(this->size_class);
  std::ostringstream oss;
  oss << "a2c-" << size_class_str << "-ibconvlsm";
  return oss.str();
}
} // namespace ai_pass_selector
