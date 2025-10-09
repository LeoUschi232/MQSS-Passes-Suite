#include "Agents/A3C/a3c_trainer.hpp"

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
#include <future>

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
train_a3c(std::unique_ptr<BaseA3CAgent> agent_boss, const std::string &dataset,
          std::unordered_map<std::string, std::string> params) {
  unsigned int max_qubits = agent_boss->getMaxQubits();
  if (params["print_param_info"] == "true") {
    std::cout << "Training A3C agent with parameters:" << std::endl;
    std::cout << "  max_qubits: " << max_qubits << std::endl;
    for (auto [key, value] : params) {
      std::cout << "  " << key << ": " << value << std::endl;
    }
  }

  // Default values
  unsigned int nr_asynchronous_agents =
      std::stoul(params["nr_asynchronous_agents"]);
  unsigned int nr_episodes = std::stoul(params["nr_episodes"]);
  unsigned int max_steps_per_episode =
      std::stoul(params["max_steps_per_episode"]);
  double discount_factor = std::stod(params["discount_factor"]);
  double gae_hyperparameter = std::stod(params["gae_hyperparameter"]);
  double entropy_coefficient = std::stod(params["entropy_coefficient"]);
  torch::Device device = DEVICE_NAME_TO_TORCH.at(params["device"]);

  if (nr_asynchronous_agents <= 0 || nr_episodes <= 0) {
    std::cerr << "Nothing to train." << std::endl;
    return {};
  }
  auto optional_statistics = get_precomputed_dataset_statistics(dataset);
  if (!optional_statistics.has_value()) {
    std::cerr << "Dataset " + dataset + " doesn't have statistics for training."
              << std::endl;
    return {};
  }
  auto [qubits_cholesky_params, gates_weights] = optional_statistics.value();

  torch::TensorOptions options =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  double global_max_reward = -std::numeric_limits<double>::max();
  int64_t T = max_steps_per_episode;

  std::vector<QuantumCircuitEnvironment> environments;
  std::vector<std::unique_ptr<BaseA3CAgent>> agents;
  environments.reserve(nr_asynchronous_agents);
  agents.reserve(nr_asynchronous_agents);
  for (unsigned int i = 0; i < nr_asynchronous_agents; i++) {
    environments.emplace_back(max_qubits, params);
    environments.back().register_randomizer_params(qubits_cholesky_params,
                                                   gates_weights);
    agents.push_back(agent_boss->clone());
  }

  std::cout << "Beginning training." << std::endl;
  updateProgress(0, nr_episodes, "Beginning training");
  for (unsigned int episode_nr = 1; episode_nr <= nr_episodes; episode_nr++) {
    if (interrupted) {
      std::cout << "\nCaught Ctrl+C Interruption in A3C training." << std::endl;
      break;
    }
    try {
      std::vector<std::future<std::tuple<torch::Tensor, torch::Tensor, double>>>
          futures;
      futures.reserve(nr_asynchronous_agents);
      for (unsigned int i = 0; i < nr_asynchronous_agents; i++) {
        futures.emplace_back(std::async(std::launch::async, [&, i] {
          QuantumCircuitEnvironment &environment = environments[i];
          BaseA3CAgent *agent = agents[i].get();
          environment.reset();
          agent->load_model();
          torch::Tensor episode_log_probs = torch::zeros({T}, options);
          torch::Tensor episode_values = torch::zeros({T + 1}, options);
          torch::Tensor episode_rewards = torch::zeros({T}, options);
          torch::Tensor episode_entropies = torch::zeros({T}, options);
          torch::Tensor termination_masks = torch::zeros({T}, options);

          for (unsigned int update_step = 0u;
               update_step < max_steps_per_episode; update_step++) {

            auto [action, log_action_probs, state_values, step_entropy] =
                agent->select_action(
                    environment.get_observation_as_torch_tensor());
            auto [reward, terminated, truncated] =
                environment.step(action.item<int>());
            episode_log_probs[update_step] = log_action_probs;
            episode_values[update_step] = state_values;
            episode_entropies[update_step] = step_entropy;
            episode_rewards[update_step] = reward;
            // Only check terminated so that bootstrapping is applied for
            // truncated but not for terminated environments.
            termination_masks[update_step] = terminated ? 0.0 : 1.0;
          }
          // Bootstrap value
          episode_values[T] =
              agent->get_value(environment.get_observation_as_torch_tensor());

          auto [actor_loss, critic_loss] = BaseA3CAgent::get_losses(
              episode_rewards, episode_log_probs, episode_values,
              episode_entropies, termination_masks, discount_factor,
              gae_hyperparameter, entropy_coefficient);

          // [actor loss, critic loss, total worker reward]
          return std::make_tuple(actor_loss, critic_loss,
                                 episode_rewards.sum().item<double>());
        }));
      }

      double episode_max_reward = -std::numeric_limits<double>::max();
      double episode_avg_reward = 0.0;
      for (unsigned int i = 0; i < nr_asynchronous_agents; i++) {
        auto [actor_loss, critic_loss, total_worker_reward] = futures[i].get();
        episode_max_reward = std::max(episode_max_reward, total_worker_reward);
        episode_avg_reward += total_worker_reward;
        agent_boss->update_parameters(actor_loss, critic_loss);
      }
      agent_boss->save_model();
      episode_avg_reward /= nr_asynchronous_agents;

      updateProgress(
          episode_nr, 1,
          "Global Max: " + std::to_string(global_max_reward) +
              " | Episode Max: " + std::to_string(episode_max_reward) +
              " | Episode Avg: " + std::to_string(episode_avg_reward));
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

  return {{"global_max_reward", std::to_string(global_max_reward)}};
}
} // namespace ai_pass_selector