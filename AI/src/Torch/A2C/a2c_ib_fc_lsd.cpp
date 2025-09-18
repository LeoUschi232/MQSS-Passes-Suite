#include "Torch/A2C/a2c_ib_fc_lsd.hpp"

#include "Environment/quantum_circuit_tensor.hpp"

#include <mlir_utils.hpp>
#include <Environment/environment.hpp>
#include <Torch/parallel_environments.hpp>
#include <Utils/circuit_utils.hpp>
#include <Utils/info_utils.hpp>
#include <Utils/passes_utils.hpp>
#include <Utils/progress_bar.hpp>
#include <stdexcept>
#include <utility>


namespace ai_pass_selector {
A2C_IB_FC_LSD::A2C_IB_FC_LSD(
    int circuit_size_class,
    std::unordered_map<std::string, std::string> params)
  : BaseA2CAgent(circuit_size_class, std::move(params)) {
  unsigned int nr_input_values
      = max_instructions * (max_qubits + NR_GATES + MAX_GATE_PARAMS);
  int critic_layer1_size = static_cast<int>(std::lround(
      std::cbrt(static_cast<double>(nr_input_values * nr_input_values))));
  int critic_layer2_size = static_cast<int>(std::lround(
      std::cbrt(static_cast<double>(nr_input_values))));
  int actor_layer1_size = static_cast<int>(std::lround(std::cbrt(
      static_cast<double>(nr_input_values * nr_input_values * NR_PASSES))));
  int actor_layer2_size = static_cast<int>(std::lround(std::cbrt(
      static_cast<double>(nr_input_values * NR_PASSES * NR_PASSES))));
  auto critic = torch::nn::Sequential(
      torch::nn::Linear(nr_input_values, critic_layer1_size),
      torch::nn::Tanh(),
      torch::nn::Linear(critic_layer1_size, critic_layer2_size),
      torch::nn::Tanh(),
      torch::nn::Linear(critic_layer2_size, 1));
  auto actor = torch::nn::Sequential(
      torch::nn::Linear(nr_input_values, actor_layer1_size),
      torch::nn::Tanh(),
      torch::nn::Linear(actor_layer1_size, actor_layer2_size),
      torch::nn::Tanh(),
      torch::nn::Linear(actor_layer2_size, NR_PASSES),
      torch::nn::Softmax(torch::nn::SoftmaxOptions(/*dim*/-1)));
  bool initialized
      = initialize(nr_input_values, critic, actor);
  if (!initialized) {
    std::cerr << "Failed to initialize A2C_IB_FC_LSD." << std::endl;
  }
}

A2C_IB_FC_LSD::A2C_IB_FC_LSD(
    const std::string &circuit_size,
    std::unordered_map<std::string, std::string> params)
  : A2C_IB_FC_LSD(
        [&circuit_size]() {
          const auto it = CIRCUIT_SIZE_TO_CLASS.find(circuit_size);
          if (it == CIRCUIT_SIZE_TO_CLASS.end()) {
            throw std::runtime_error("Unsupported size: " + circuit_size);
          }
          return it->second;
        }(),
        std::move(params)) {}

std::string A2C_IB_FC_LSD::agentName() const {
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
  oss << "a2c-" << size_class_str << "-ib-fc-lsd";
  return oss.str();
}
} // namespace ai_pass_selector