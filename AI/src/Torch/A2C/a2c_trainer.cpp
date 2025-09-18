#include "Torch/A2C/a2c_trainer.hpp"

#include <Quake.hpp>
#include <filesystem>
#include <mlir_utils.hpp>
#include <Torch/parallel_environments.hpp>
#include <Utils/info_utils.hpp>
#include <Utils/progress_bar.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>

using namespace mqss::support::quakeDialect;
namespace fs = std::filesystem;

namespace ai_pass_selector {
A2CTrainerConfig build_a2c_trainer_config(
    const std::unordered_map<std::string, std::string> &params) {
  A2CTrainerConfig config;
  for (const auto &[key, value] : params) {
    if (key == "nr_parallel_environments") {
      config.nr_parallel_environments = std::stoul(value);
      config.applied_overrides.push_back(key + "=" + value);
    } else if (key == "episodes") {
      config.episodes = std::stoul(value);
      config.applied_overrides.push_back(key + "=" + value);
    } else if (key == "max_steps_per_episode") {
      config.max_steps_per_episode = std::stoul(value);
      config.applied_overrides.push_back(key + "=" + value);
    } else if (key == "discount_factor") {
      config.discount_factor = std::stod(value);
      config.applied_overrides.push_back(key + "=" + value);
    } else if (key == "gae_hyperparameter") {
      config.gae_hyperparameter = std::stod(value);
      config.applied_overrides.push_back(key + "=" + value);
    } else if (key == "entropy_coefficient") {
      config.entropy_coefficient = std::stod(value);
      config.applied_overrides.push_back(key + "=" + value);
    } else if (key == "device") {
      std::string value_lower = value;
      std::transform(
          value_lower.begin(), value_lower.end(), value_lower.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (value_lower == "cpu") {
        config.device = torch::kCPU;
        config.applied_overrides.push_back("device=cpu");
      } else if (value_lower == "cuda" || value_lower == "gpu") {
        if (torch::cuda::is_available()) {
          config.device = torch::kCUDA;
          config.applied_overrides.push_back("device=cuda");
        } else {
          config.applied_overrides.push_back("device=" + value_lower + " (unavailable)");
        }
      }
    }
  }
  return config;
}

std::unordered_map<std::string, std::string> train_a2c(
    BaseA2CAgent &agent,
    const std::string &dataset,
    std::unordered_map<std::string, std::string> params) {
  unsigned int max_qubits = agent.getMaxQubits();
  unsigned int max_instructions = agent.getMaxInstructions();
  unsigned int max_depth = agent.getMaxDepth();

  A2CTrainerConfig config = build_a2c_trainer_config(params);
  if (!config.applied_overrides.empty()) {
    std::ostringstream overrides_stream;
    overrides_stream << "train_a2c overrides:";
    for (const auto &override_entry : config.applied_overrides) {
      overrides_stream << ' ' << override_entry;
    }
    std::cout << overrides_stream.str() << std::endl;
  }
  unsigned int nr_parallel_environments = config.nr_parallel_environments;
  unsigned int episodes = config.episodes;
  unsigned int max_steps_per_episode = config.max_steps_per_episode;
  double discount_factor = config.discount_factor;
  double gae_hyperparameter = config.gae_hyperparameter;
  double entropy_coefficient = config.entropy_coefficient;
  torch::Device device = config.device;

  if (agent.getNrInputValues() <= 0 || nr_parallel_environments <= 0) {
    std::cerr << "No agent to train." << std::endl;
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
  ParallelEnvironments environments(
      nr_parallel_environments, max_qubits, max_instructions, max_depth,
      max_steps_per_episode);

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
  return {
      {"max_reward", std::to_string(max_reward)},
      {"average_reward", std::to_string(average_reward)},
      {"final_entropy", std::to_string(entropies.back())},
      {"final_critic_loss", std::to_string(critic_losses.back())},
      {"final_actor_loss", std::to_string(actor_losses.back())}
  };
}
} // namespace ai_pass_selector