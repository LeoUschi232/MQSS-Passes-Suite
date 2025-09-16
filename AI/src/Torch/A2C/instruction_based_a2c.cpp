#include "Torch/A2C/instruction_based_a2c.hpp"

#include "Environment/quantum_circuit_tensor.hpp"

#include <mlir_utils.hpp>
#include <Environment/environment.hpp>
#include <Utils/info_utils.hpp>
#include <Utils/passes_utils.hpp>
#include <Utils/progress_bar.hpp>


namespace ai_pass_selector {
std::unordered_map<std::string, std::string> InstructionBasedA2CAgent::train(
    std::string dataset,
    unsigned int episodes,
    double discount_factor,
    double gae_hyperparameter,
    double entropy_coefficient,
    unsigned int max_steps_per_episode) {
  if (nr_input_values <= 0 || nr_parallel_environments <= 0) {
    std::cerr << "No agent to train." << std::endl;
    return {};
  }
  if (fs::path dataset_dir = fs::path(AI_DATASET_DIR) / "Quake" / dataset;
    !fs::exists(dataset_dir) || !fs::is_directory(dataset_dir)) {
    std::cerr << "Dataset: " << dataset << " not found." << std::endl;
    return {};
  }
  std::vector<fs::path> all_dataset_files = get_dataset_files(dataset);
  std::vector<std::string> filtered_dataset_files;
  for (auto &file : all_dataset_files) {
    std::string quake_module_text = readFileToString(file.string());
    if (auto [mlir_module, context_ptr] = extractMLIRContext(quake_module_text);
      getNumberOfQubits(FuncOp(mlir_module)) > max_qubits
      || getNumberOfGates(FuncOp(mlir_module)) > max_instructions) {
      continue;
    }
    filtered_dataset_files.push_back(file.string());
  }
  unsigned int dataset_size = filtered_dataset_files.size();
  if (dataset_size <= 0) {
    std::cerr << "No dataset files found." << std::endl;
    return {};
  }

  std::vector<QuantumCircuitEnviorment> environments;

  double max_reward = -std::numeric_limits<double>::max();
  double average_reward = 0.0;
  std::vector<double> entropies;
  std::vector<double> critic_losses;
  std::vector<double> actor_losses;
  for (unsigned int episode_nr = 1; episode_nr <= episodes; episode_nr++) {
    updateProgress(episode_nr, episodes,
                   "Episode " + std::to_string(episode_nr));
    environments.clear();
    environments.reserve(nr_parallel_environments);
    for (unsigned int i = 0; i < nr_parallel_environments; i++) {
      fs::path random_dataset_entry
          = filtered_dataset_files[random_int(0, dataset_size)];
      environments.emplace_back(
          max_qubits, max_instructions, max_depth,
          random_dataset_entry, max_steps_per_episode);
    }
    if (environments.size() > nr_parallel_environments) {
      throw std::runtime_error("Emplace_back doesn't work as expected.");
    }

    auto episode_log_probs = torch::zeros(
        {nr_parallel_environments, NR_PASSES}, device);
    auto episode_values = torch::zeros(
        {nr_parallel_environments, NR_PASSES}, device);
    auto episode_rewards = torch::zeros(
        {nr_parallel_environments, NR_PASSES}, device);
    auto termination_masks = torch::zeros(
        {nr_parallel_environments, NR_PASSES}, device);

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

std::string IB_FC_LSM_A2C::model_name() const {
  std::ostringstream oss;
  oss << "ib-fc-lsm-a2c-"
      << max_qubits << "x" << max_instructions << "x" << max_depth;
  return oss.str();
}

} // namespace ai_pass_selector