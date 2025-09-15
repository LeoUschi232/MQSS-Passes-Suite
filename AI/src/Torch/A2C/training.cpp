#include "Torch/training.hpp"

// Utils includes
#include "Utils/progress_bar.hpp"

// Stdandard library includes
#include <unordered_map>
#include <string>
#include <Environment/environment.hpp>
#include <Torch/agent_utils..hpp>

namespace fs = std::filesystem;

namespace ai_pass_selector {

std::unordered_map<std::string, std::string> train_a2c(
    const torch::nn::Module &agent,
    unsigned int nr_parallel_environments,
    unsigned int episodes,
    double discount_factor,
    double gae_hyperparameter,
    double entropy_coefficient,
    torch::Device device,
    std::string dataset) {
  if (nr_parallel_environments <= 0) {
    std::cerr << "No environments to train on." << std::endl;
    return {};
  }
  fs::path dataset_dir = fs::path(AI_DATASET_DIR) / "Quake" / dataset;
  if (!fs::exists(dataset_dir) || !fs::is_directory(dataset_dir)) {
    std::cerr << "Dataset: " << dataset << " not found." << std::endl;
    return {};
  }
  unsigned int nr_actions = getNrOfPasses();
  std::vector<QuantumCircuitEnviorment> environments;
  environments.reserve(nr_parallel_environments);

  double max_reward = -std::numeric_limits<double>::max();
  double average_reward = 0.0;
  std::vector<double> entropies;
  std::vector<double> critic_losses;
  std::vector<double> actor_losses;

  for (unsigned int episode_nr = 1; episode_nr <= episodes; episode_nr++) {
    auto episode_log_probs = torch::zeros(
        {nr_parallel_environments, nr_actions}, device);
    auto episode_values = torch::zeros(
        {nr_parallel_environments, nr_actions}, device);
    auto episode_rewards = torch::zeros(
        {nr_parallel_environments, nr_actions}, device);
    auto termination_masks = torch::zeros(
        {nr_parallel_environments, nr_actions}, device);

    double entropy = 0.0;

  }

  return {};
}
} // namespace ai_pass_selector