#include "Torch/A2C/a2c_ib_fc_lsm.hpp"

// Environment includes
#include <Environment/environment.hpp>

// Utils includes
#include <Torch/parallel_environments.hpp>
#include <Utils/circuit_utils.hpp>
#include <Utils/passes_utils.hpp>


namespace ai_pass_selector {
A2C_IB_FC_LSM::A2C_IB_FC_LSM(
    int circuit_size_class,
    std::unordered_map<std::string, std::string> params)
  : BaseA2CAgent(circuit_size_class, std::move(params)) {
  unsigned int nr_input_values
      = max_instructions * (max_qubits + NR_GATES + MAX_GATE_PARAMS);
  auto critic = torch::nn::Sequential(
      torch::nn::Linear(nr_input_values, nr_input_values),
      torch::nn::Tanh(),
      torch::nn::Linear(nr_input_values, nr_input_values),
      torch::nn::Tanh(),
      torch::nn::Linear(nr_input_values, 1));
  auto actor = torch::nn::Sequential(
      torch::nn::Linear(nr_input_values, nr_input_values),
      torch::nn::Tanh(),
      torch::nn::Linear(nr_input_values, nr_input_values),
      torch::nn::Tanh(),
      torch::nn::Linear(nr_input_values, NR_PASSES),
      torch::nn::Softmax(torch::nn::SoftmaxOptions(/*dim*/-1)));
  bool initialized
      = initialize(nr_input_values, critic, actor);
  if (!initialized) {
    std::cerr << "Failed to initialize IB_FC_LSD_A2C." << std::endl;
  }
}


std::string A2C_IB_FC_LSM::agentName() const {
  std::string size_class_str;
  switch (this->size_class) {
  case TINY:
    size_class_str = "tiny";
    break;
  case SMALL:
    size_class_str = "small";
    break;
  case MODERATE:
    size_class_str = "moderate";
    break;
  case BIG:
    size_class_str = "big";
    break;
  case HUGE:
    size_class_str = "huge";
    break;
  default:
    throw std::runtime_error("Unsupported size class.");
  }
  std::ostringstream oss;
  oss << "a2c-" << size_class_str << "-ibfclsm";
  return oss.str();
}
} // namespace ai_pass_selector