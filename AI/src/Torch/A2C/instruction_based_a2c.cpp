#include "Torch/A2C/instruction_based_a2c.hpp"

#include "Environment/quantum_circuit_tensor.hpp"

#include <mlir_utils.hpp>
#include <Environment/environment.hpp>
#include <Utils/info_utils.hpp>


namespace ai_pass_selector {
std::unordered_map<std::string, std::string> InstructionBasedA2CAgent::train(
    std::string dataset,
    unsigned int episodes,
    double discount_factor,
    double gae_hyperparameter,
    double entropy_coefficient) {
  if (nr_input_values <= 0 || nr_parallel_environments <= 0) {
    std::cerr << "No agent to train." << std::endl;
    return {};
  }
  if (fs::path dataset_dir = fs::path(AI_DATASET_DIR) / "Quake" / dataset;
    !fs::exists(dataset_dir) || !fs::is_directory(dataset_dir)) {
    std::cerr << "Dataset: " << dataset << " not found." << std::endl;
    return {};
  }
  std::vector<fs::path> dataset_files = get_dataset_files(dataset);
  if (dataset_files.empty()) {
    std::cerr << "Dataset: " << dataset << " not found." << std::endl;
    return {};
  }
  auto [dataset_size,
        dataset_min_qubits, dataset_avg_qubits, dataset_max_qubits,
        dataset_min_gates, dataset_avg_gates, dataset_max_gates,
        dataset_min_depth, dataset_avg_depth, dataset_max_depth]
      = get_dataset_info(dataset).value();
  if (dataset_min_qubits > max_qubits || dataset_min_gates > max_instructions) {
    std::cerr << "Dataset: " << dataset << " incompatible with agent: "
        << model_name() << std::endl;
    return {};
  }

  std::vector<QuantumCircuitEnviorment> environments;
    environments.reserve(nr_parallel_environments);

  double max_reward = -std::numeric_limits<double>::max();
  double average_reward = 0.0;
  std::vector<double> entropies;
  std::vector<double> critic_losses;
  std::vector<double> actor_losses;
  for (unsigned int episode_nr = 1; episode_nr <= episodes; episode_nr++) {

    environments.clear();
    for (unsigned int i = 0; i < nr_parallel_environments; i++) {
      while (true) {
        fs::path random_dataset_entry
            = dataset_files[random_int(0, dataset_size)];
        std::string quake_module_text
            = readFileToString(random_dataset_entry.string());
        auto [mlir_module, context_ptr] = extractMLIRContext(quake_module_text);
        if (getNumberOfQubits(FuncOp(mlir_module)) > max_qubits
            || getNumberOfGates(FuncOp(mlir_module)) > max_instructions) {
          continue;
            }
        environments.emplace_back(max_qubits, max_instructions, max_depth, mlir_module);
        break;
      }
    }
    if (environments.size() > nr_parallel_environments) {
      throw std::runtime_error("Emplace_back doesn't work as expected.");
    }


    auto episode_log_probs = torch::zeros(
        {nr_parallel_environments, nr_output_values}, device);
    auto episode_values = torch::zeros(
        {nr_parallel_environments, nr_output_values}, device);
    auto episode_rewards = torch::zeros(
        {nr_parallel_environments, nr_output_values}, device);
    auto termination_masks = torch::zeros(
        {nr_parallel_environments, nr_output_values}, device);

    double entropy = 0.0;

  }
  return {};
}

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
  : InstructionBasedA2CAgent(
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
  : InstructionBasedA2CAgent(
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