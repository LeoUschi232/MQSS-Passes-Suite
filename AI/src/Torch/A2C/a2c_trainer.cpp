#include "Torch/A2C/a2c_trainer.hpp"

#include <Quake.hpp>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mlir_utils.hpp>
#include <Torch/parallel_environments.hpp>
#include <Utils/info_utils.hpp>
#include <Utils/progress_bar.hpp>

using namespace mqss::support::quakeDialect;
namespace fs = std::filesystem;

namespace ai_pass_selector {
std::unordered_map<std::string, std::string> train_a2c(
    BaseA2CAgent &agent,
    const std::string &dataset,
    std::unordered_map<std::string, std::string> params) {
  unsigned int max_qubits = agent.getMaxQubits();
  unsigned int max_instructions = agent.getMaxInstructions();
  unsigned int max_depth = agent.getMaxDepth();

  // Default values
  unsigned int nr_parallel_environments = 10;
  unsigned int episodes = 1000;
  unsigned int max_steps_per_episode = 20;
  double discount_factor = 1.0;
  double gae_hyperparameter = 0.96;
  double entropy_coefficient = 0.01;
  torch::Device device = torch::kCPU;
  for (auto [key, value] : params) {
    if (key == "nr_parallel_environments") {
      nr_parallel_environments = std::stoul(value);
    } else if (key == "episodes") {
      episodes = std::stoul(value);
    } else if (key == "max_steps_per_episode") {
      max_steps_per_episode = std::stoul(value);
    } else if (key == "discount_factor") {
      discount_factor = std::stod(value);
    } else if (key == "gae_hyperparameter") {
      gae_hyperparameter = std::stod(value);
    } else if (key == "entropy_coefficient") {
      entropy_coefficient = std::stod(value);
    } else if (key == "device"
               && (value == "cuda" || value == "gpu")
               && torch::cuda::is_available()) {
      device = torch::kCUDA;
    }
  }

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
    std::vector<fs::path> environment_circuits(nr_parallel_environments);
    for (unsigned int i = 0; i < nr_parallel_environments; i++) {
      fs::path random_dataset_entry
          = filtered_dataset_files[random_int(0, dataset_size)];
      environment_circuits[i] = random_dataset_entry;
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
    std::vector<uint8_t> active_mask(B, 1);
    unsigned int active_envs = B;
    for (unsigned int update_step = 0;
         update_step < max_steps_per_episode;
         update_step++) {
      if (active_envs == 0) {
        break;
      }
      std::vector<uint8_t> step_mask = active_mask;
      auto [actions, log_action_probs, state_values, step_entropy]
          = agent.select_action(batched_observations);
      for (int64_t b = 0; b < B; ++b) {
        if (step_mask[b]) {
          continue;
        }
        actions[b] = 0;
        const int64_t index = b;
        log_action_probs.index_put_({index}, 0.0);
        state_values.index_put_({index}, 0.0);
        step_entropy.index_put_({index}, 0.0);
      }
      auto [rewards, terminates] = environments.step(actions, step_mask);
      episode_log_probs[update_step] = log_action_probs;
      episode_values[update_step] = state_values;
      episode_entropies[update_step] = step_entropy;
      for (int64_t b = 0; b < B; b++) {
        episode_rewards[update_step][b] = step_mask[b] ? rewards[b] : 0.0;
        termination_masks[update_step][b] = (step_mask[b] && !terminates[b]) ? 1.0 : 0.0;
        if (step_mask[b] && terminates[b]) {
          active_mask[b] = 0;
          if (active_envs > 0) {
            active_envs--;
          }
          if (!environment_circuits[b].empty()
              && !environments.register_quantum_circuit(
                  static_cast<unsigned int>(b),
                  environment_circuits[b])) {
            std::cerr << "Failed to reset circuit for environment " << b
                      << std::endl;
          }
        }
      }
      if (active_envs == 0) {
        break;
      }
      batched_observations
          = environments.get_batched_instruction_based_observations();
      for (int64_t b = 0; b < B; ++b) {
        if (!active_mask[b]) {
          batched_observations.select(0, b).zero_();
        }
      }
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