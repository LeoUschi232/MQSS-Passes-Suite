#include "Torch/A2C/a2c_ib_fc_lsm.hpp"

// Environment includes
#include <Environment/environment.hpp>

// Utils includes
#include <Torch/parallel_environments.hpp>
#include <Utils/circuit_utils.hpp>
#include <Utils/info_utils.hpp>
#include <Utils/passes_utils.hpp>
#include <Utils/progress_bar.hpp>


namespace ai_pass_selector {
A2C_IB_FC_LSM::A2C_IB_FC_LSM(
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
  std::ostringstream oss;
  oss << "a2c-ib-fc-lsm-"
      << max_qubits << "x" << max_instructions << "x" << max_depth;
  return oss.str();
}


std::unordered_map<std::string, std::string> train_agent(
    A2C_IB_FC_LSM &agent,
    std::string dataset,
    unsigned int episodes,
    double discount_factor,
    double gae_hyperparameter,
    double entropy_coefficient,
    unsigned int max_steps_per_episode) {

  unsigned int max_qubits = agent.getMaxQubits();
  unsigned int max_instructions = agent.getMaxInstructions();
  unsigned int max_depth = agent.getMaxDepth();
  unsigned int nr_parallel_environments = agent.getNrParallelEnvironments();
  torch::Device device = agent.getDevice();
  if (unsigned int nr_input_values = agent.getNrInputValues();
    nr_input_values <= 0 || nr_parallel_environments <= 0) {
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
    auto [mlir_module, context_ptr] = extractMLIRContext(quake_module_text);
    auto kernel = getKernelEntryPoint(mlir_module);
    if (!kernel) {
      continue;
    }
    if (getNumberOfQubits(kernel) > max_qubits
        || getNumberOfGates(kernel) > max_instructions) {
      continue;
    }
    filtered_dataset_files.push_back(file.string());
  }
  unsigned int dataset_size = filtered_dataset_files.size();
  if (dataset_size <= 0) {
    std::cerr << "No dataset files found." << std::endl;
    return {};
  }
  ParallelEnvironments environments(
      nr_parallel_environments, max_qubits, max_instructions, max_depth,
      max_steps_per_episode);
  agent.setNrParallelEnvironments(nr_parallel_environments);

  double max_reward = -std::numeric_limits<double>::max();
  double average_reward = 0.0;
  std::vector<double> entropies;
  std::vector<double> critic_losses;
  std::vector<double> actor_losses;

  for (unsigned int episode_nr = 1; episode_nr <= episodes; episode_nr++) {

    for (unsigned int i = 0; i < nr_parallel_environments; i++) {
      fs::path random_dataset_entry
          = filtered_dataset_files[random_int(0, dataset_size)];
      environments.register_quantum_circuit(i, random_dataset_entry);
    }
    int64_t T = max_steps_per_episode;
    int64_t B = nr_parallel_environments;
    torch::TensorOptions options = torch::TensorOptions().device(device).dtype(
        torch::kFloat64);
    auto episode_log_probs = torch::zeros({T, B}, options);
    auto episode_values = torch::zeros({T, B}, options);
    auto episode_rewards = torch::zeros({T, B}, options);
    auto episode_entropies = torch::zeros({T, B}, options);
    auto termination_masks = torch::zeros({T, B}, options);

    torch::Tensor batched_observations =
        environments.get_batched_instruction_based_observations();
    for (unsigned int update_step = 0;
         update_step < max_steps_per_episode;
         update_step++) {
      auto [actions, log_action_probs, state_values, step_entropy]
          = agent.select_action(batched_observations);
      auto [rewards, terminates] = environments.step(actions);
      episode_log_probs[update_step] = log_action_probs;
      episode_values[update_step] = state_values;
      episode_entropies[update_step] = step_entropy;
      for (unsigned int b = 0; b < B; b++) {
        episode_rewards[update_step][b] = rewards[b];
        termination_masks[update_step][b] = terminates[b] ? 0.0 : 1.0;
      }
      batched_observations
          = environments.get_batched_instruction_based_observations();
    }

    auto [critic_loss, actor_loss] = agent.get_losses(
        episode_rewards,
        episode_log_probs,
        episode_values,
        episode_entropies,
        termination_masks,
        discount_factor,
        gae_hyperparameter,
        entropy_coefficient);

    auto episode_rewards_cpu = episode_rewards.to(torch::kCPU);
    auto total_rewards = episode_rewards_cpu.sum(/*axis=*/0);
    if (total_rewards.size(/*dim=*/0) != nr_parallel_environments) {
      throw std::runtime_error("total_rewards.size=/=nr_parallel_environments");
    }
    average_reward = total_rewards.mean().item<double>();
    if (average_reward > max_reward) {
      max_reward = average_reward;
      agent.save_model();
    }
    agent.update_parameters(critic_loss, actor_loss);
    entropies.push_back(episode_entropies.mean().item<double>());
    critic_losses.push_back(critic_loss.item<double>());
    actor_losses.push_back(actor_loss.item<double>());
    updateProgress(
        episode_nr, episodes,
        "Max: " + std::to_string(max_reward)
        + " | Avg: " + std::to_string(average_reward)
        + " | Critic: " + std::to_string(critic_loss.item<double>())
        + " | Actor: " + std::to_string(actor_loss.item<double>()));
  }
  return {};
}

} // namespace ai_pass_selector