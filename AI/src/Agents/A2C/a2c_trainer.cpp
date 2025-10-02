#include "Agents/A2C/a2c_trainer.hpp"

// Environment includes
#include "Environment/parallel_environments.hpp"

// Agents includes
#include "Agents/agent_utils.hpp"

// Utils includes
#include "Support/mlir_utils.hpp"
#include "Utils/info_utils.hpp"
#include "Utils/progress_bar.hpp"

// Standard library includes
#include <csignal>
#include <filesystem>

using namespace mqss::support::quakeDialect;
namespace fs = std::filesystem;

// Global flag for SIGINT Ctrl+C interruptions.
volatile sig_atomic_t interrupted = 0;
void signal_handler(int signal) {
  if (signal == SIGINT) {
    interrupted = 1;
  }
}

namespace ai_pass_selector {
std::unordered_map<std::string, std::string>
train_a2c(std::unique_ptr<BaseA2CAgent> agent, const std::string &dataset,
          std::unordered_map<std::string, std::string> params) {
  unsigned int max_qubits = agent->getMaxQubits();
  if (params["print_param_info"] == "true") {
    std::cout << "Training A2C agent with parameters:" << std::endl;
    std::cout << "  max_qubits: " << max_qubits << std::endl;
    for (auto [key, value] : params) {
      std::cout << "  " << key << ": " << value << std::endl;
    }
  }

  // Default values
  unsigned int nr_parallel_environments =
      std::stoul(params["nr_parallel_environments"]);
  unsigned int episodes = std::stoul(params["episodes"]);
  unsigned int max_steps_per_episode =
      std::stoul(params["max_steps_per_episode"]);
  double discount_factor = std::stod(params["discount_factor"]);
  double gae_hyperparameter = std::stod(params["gae_hyperparameter"]);
  double entropy_coefficient = std::stod(params["entropy_coefficient"]);
  torch::Device device = DEVICE_NAME_TO_TORCH.at(params["device"]);

  if (nr_parallel_environments <= 0) {
    std::cerr << "No agent to train." << std::endl;
    return {};
  }
  auto optional_statistics = get_precomputed_dataset_statistics(dataset);
  if (!optional_statistics.has_value()) {
    std::cerr << "Dataset " + dataset + " doesn't have statistics for training."
              << std::endl;
    return {};
  }
  auto [qubits_cholesky_params, gates_weights] = optional_statistics.value();
  ParallelEnvironments environments(nr_parallel_environments, max_qubits,
                                    max_steps_per_episode);
  environments.register_randomizer_params(qubits_cholesky_params,
                                          gates_weights);

  double global_max_reward = -std::numeric_limits<double>::max();
  int64_t T = max_steps_per_episode;
  int64_t B = nr_parallel_environments;

  std::cout << "Beginning training." << std::endl;
  updateProgress(0, episodes, "Beginning training");
  for (unsigned int episode_nr = 1; episode_nr <= episodes; episode_nr++) {
    if (interrupted) {
      std::cout << "\nCaught Ctrl+C Interruption in A2c training." << std::endl;
      break;
    }
    double local_max_reward = -std::numeric_limits<double>::max();
    try {
      auto [success, nr_qubits, nr_gates] =
          environments.randomize_all_circuits_with_equal_dimensions();
      torch::TensorOptions options =
          torch::TensorOptions().device(device).dtype(torch::kFloat64);
      torch::Tensor episode_log_probs = torch::zeros({T, B}, options);
      torch::Tensor episode_values = torch::zeros({T + 1, B}, options);
      torch::Tensor episode_rewards = torch::zeros({T, B}, options);
      torch::Tensor episode_entropies = torch::zeros({T, B}, options);
      torch::Tensor termination_masks = torch::zeros({T, B}, options);

      for (unsigned int update_step = 0u; update_step < max_steps_per_episode;
           update_step++) {

        auto [batched_observations, padding_on_observations] =
            environments.get_batched_observations_with_padding();

        auto [actions, log_action_probs, state_values, step_entropy] =
            agent->select_action(batched_observations);
        std::vector<std::tuple<double, bool, bool>> step_returns =
            environments.step(actions);

        episode_log_probs[update_step] = log_action_probs;
        episode_values[update_step] = state_values;
        episode_entropies[update_step] = step_entropy;
        for (unsigned int b = 0; b < B; b++) {
          auto [reward, terminated, truncated] = step_returns[b];
          episode_rewards[update_step][b] = reward;
          termination_masks[update_step][b] =
              terminated || truncated ? 0.0 : 1.0;
        }
      }
      // Bootstrap value
      episode_values[T] =
          agent->get_value(environments.get_batched_observations());

      auto [actor_loss, critic_loss] = agent->get_losses(
          episode_rewards, episode_log_probs, episode_values, episode_entropies,
          termination_masks, discount_factor, gae_hyperparameter,
          entropy_coefficient);

      auto episode_rewards_cpu = episode_rewards.to(torch::kCPU);
      auto total_rewards = episode_rewards_cpu.sum(/*axis=*/0);
      if (total_rewards.size(/*dim=*/0) != nr_parallel_environments) {
        throw std::runtime_error(
            "total_rewards.size=/=nr_parallel_environments");
      }
      double current_reward = total_rewards.sum().item<double>();
      summed_rewards += current_reward;
      if (current_reward > max_reward) {
        max_reward = current_reward;
        agent->save_model();
      }
      agent->update_parameters(critic_loss, actor_loss);
      updateProgress(
          episode_nr, episodes,
          "Global Max: " + std::to_string(max_reward) +
              " | Local Max: " + std::to_string(summed_rewards / episode_nr) +
              " | Avg: " + std::to_string(summed_rewards / episode_nr) +
              " | Nr qubits: " + std::to_string(nr_qubits) +
              " | Nr gates: " + std::to_string(nr_gates)

      );
    } catch (const std::exception &e) {
      std::cerr << "\nEpisode " << episode_nr << ": " << e.what() << std::endl;
      if (params["stop_training_on_error"] == "true") {
        interrupted = 1;
        break;
      }
    }
  }
  if (!interrupted) {
    std::cout << "\nTraining finished." << std::endl;
  }

  if (params["save_agent_at_end_of_training"] == "true") {
    agent->save_model();
    std::cout << "Saved: " << agent->agentName() << std::endl;
  }
  return {{"max_reward", std::to_string(max_reward)},
          {"average_reward", std::to_string(summed_rewards / episodes)}};
}
} // namespace ai_pass_selector