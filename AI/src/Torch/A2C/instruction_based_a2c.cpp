#include "Torch/A2C/instruction_based_a2c.hpp"

#include "Environment/quantum_circuit_tensor.hpp"


namespace ai_pass_selector {

IB_FC_LSD_A2C::IB_FC_LSD_A2C(
    unsigned int max_qubits,
    unsigned int max_instructions,
    unsigned int max_depth,
    int critic_optimizer_type,
    int actor_optimizer_type,
    double critic_learning_rate,
    double actor_learning_rate,
    unsigned int nr_parallel_environments,
    torch::Device device)
  : BaseA2CAgent(
      max_qubits, max_instructions, max_depth,
      critic_optimizer_type, actor_optimizer_type,
      critic_learning_rate, actor_learning_rate,
      nr_parallel_environments, device) {
  unsigned int nr_input_values = getNrOfInputValuesForInstructionBased(
      max_qubits, max_instructions);
  unsigned int nr_actions = getNrOfPasses();
  int critic_layer1_size = static_cast<int>(std::lround(
      std::cbrt(static_cast<double>(nr_input_values * nr_input_values))));
  int critic_layer2_size = static_cast<int>(std::lround(
      std::cbrt(static_cast<double>(nr_input_values))));
  int actor_layer1_size = static_cast<int>(std::lround(std::cbrt(
      static_cast<double>(nr_input_values * nr_input_values * nr_actions))));
  int actor_layer2_size = static_cast<int>(std::lround(std::cbrt(
      static_cast<double>(nr_input_values * nr_actions * nr_actions))));
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
      torch::nn::Linear(actor_layer2_size, nr_actions),
      torch::nn::Softmax(torch::nn::SoftmaxOptions(/*dim*/-1)));
  bool initialized
      = initialize(nr_input_values, nr_actions, critic, actor);
  if (!initialized) {
    std::cerr << "Failed to initialize IB_FC_LSD_A2C." << std::endl;
  }
}

std::string IB_FC_LSD_A2C::model_name() const {
  std::ostringstream oss;
  oss << "ib-fc-lsd-a2c-"
      << max_qubits << "x" << max_instructions << "x" << max_depth;
  return oss.str();
}

IB_FC_LSM_A2C::IB_FC_LSM_A2C(
    unsigned int max_qubits,
    unsigned int max_instructions,
    unsigned int max_depth,
    int critic_optimizer_type,
    int actor_optimizer_type,
    double critic_learning_rate,
    double actor_learning_rate,
    unsigned int nr_parallel_environments,
    torch::Device device)
  : BaseA2CAgent(
      max_qubits, max_instructions, max_depth,
      critic_optimizer_type, actor_optimizer_type,
      critic_learning_rate, actor_learning_rate,
      nr_parallel_environments, device) {
  unsigned int nr_input_values = getNrOfInputValuesForInstructionBased(
      max_qubits, max_instructions);
  unsigned int nr_actions = getNrOfPasses();
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
      torch::nn::Linear(nr_input_values, nr_actions),
      torch::nn::Softmax(torch::nn::SoftmaxOptions(/*dim*/-1)));
  bool initialized
      = initialize(nr_input_values, nr_actions, critic, actor);
  if (!initialized) {
    std::cerr << "Failed to initialize IB_FC_LSD_A2C." << std::endl;
  }
}

std::string IB_FC_LSM_A2C::model_name() const {
  std::ostringstream oss;
  oss << "ib-fc-lsm-a2c-"
      << max_qubits << "x" << max_instructions << "x" << max_depth;
  return oss.str();
}

} // namespace ai_pass_selector