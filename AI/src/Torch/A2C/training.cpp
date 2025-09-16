#include "Torch/A2C/training.hpp"

// Utils includes
#include "Utils/progress_bar.hpp"

// Stdandard library includes
#include <mlir_utils.hpp>
#include <unordered_map>
#include <string>
#include <Environment/environment.hpp>
#include <Utils/info_utils.hpp>
#include <Utils/passes_utils.hpp>

namespace fs = std::filesystem;

namespace ai_pass_selector {

std::unordered_map<std::string, std::string> train(
    const BaseA2CAgent &agent,
    std::string dataset,
    unsigned int episodes,
    double discount_factor,
    double gae_hyperparameter,
    double entropy_coefficient,
    unsigned int max_steps_per_episode,
    torch::Device device) {
  unsigned int max_qubits = agent.getMaxQubits();
  unsigned int max_instructions = agent.getMaxInstructions();
  unsigned int max_depth = agent.getMaxDepth();
  unsigned int nr_input_values = agent.getNrInputValues();
  unsigned int nr_parallel_environments = agent.getNrParallelEnvironments();
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
  environments.reserve(nr_parallel_environments);

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

    {
      // TODO: Parallelize this scope
      for (unsigned int i = 0; i < nr_parallel_environments; i++) {
        fs::path random_dataset_entry
            = filtered_dataset_files[random_int(0, dataset_size)];
        environments.emplace_back(
            max_qubits, max_instructions, max_depth,
            random_dataset_entry, max_steps_per_episode);
      }
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
} // namespace ai_pass_selector