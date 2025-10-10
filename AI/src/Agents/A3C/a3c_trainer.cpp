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
train_a3c(const std::unique_ptr<BaseA3CAgent> &agent_boss,
          const std::string &dataset,
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
  unsigned int a3c_max_async_steps = std::stoul(params["a3c_max_async_steps"]);
  unsigned int max_steps_per_episode =
      std::stoul(params["max_steps_per_episode"]);
  double discount_factor = std::stod(params["discount_factor"]);
  double gae_hyperparameter = std::stod(params["gae_hyperparameter"]);
  double entropy_coefficient = std::stod(params["entropy_coefficient"]);
  torch::Device device = DEVICE_NAME_TO_TORCH.at(params["device"]);

  if (nr_asynchronous_agents <= 0 || a3c_max_async_steps <= 0) {
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

  auto global_mutex = std::make_unique<std::mutex>();
  unsigned int global_async_step = 0u;
  std::cout << "Beginning training." << std::endl;
  updateProgress(0u, a3c_max_async_steps, "Beginning training");

  std::vector<std::future<void>> futures;
  futures.reserve(nr_asynchronous_agents);
  for (unsigned int i = 0u; i < nr_asynchronous_agents; i++) {
    futures.emplace_back(std::async(std::launch::async, [&] {
      QuantumCircuitEnvironment environment(max_qubits, params);
      environment.register_randomizer_params(qubits_cholesky_params,
                                             gates_weights);
      std::unique_ptr<BaseA3CAgent> agent = agent_boss->clone();

      // A3C doesn't have a max cap on the nr of qpisodes.
      // The cap is implicit on the nr of total steps performed by all
      // agents at the same time.
      while (true) {
        {
          std::lock_guard lock(*global_mutex);
          if (global_async_step >= a3c_max_async_steps) {
            break;
          }
        }
        if (interrupted) {
          std::cout << "\nCaught Ctrl+C Interruption in A3C training."
                    << std::endl;
          break;
        }
        environment.reset();
        agent->zero_grad();
        // Load params uses the inner mutex from both agents, so no need to
        // apply the global mutex here.
        agent->load_params(*agent_boss);

        std::vector<torch::Tensor> episode_log_probs_vector;
        std::vector<torch::Tensor> episode_values_vector;
        std::vector<torch::Tensor> episode_rewards_vector;
        std::vector<torch::Tensor> episode_entropies_vector;
        episode_log_probs_vector.reserve(T);
        episode_values_vector.reserve(T + 1);
        episode_rewards_vector.reserve(T);
        episode_entropies_vector.reserve(T);

        double total_worker_reward = 0.0;
        unsigned int update_step;
        bool add_bootstrap = false;
        for (update_step = 0u; update_step < max_steps_per_episode;
             update_step++) {

          auto [action, log_action_probs, state_values, step_entropy] =
              agent->select_action(
                  environment.get_observation_as_torch_tensor());
          auto [reward, terminated, truncated] =
              environment.step(action.item<int>());
          total_worker_reward += reward;

          episode_log_probs_vector.push_back(log_action_probs);
          episode_values_vector.push_back(state_values);
          episode_entropies_vector.push_back(step_entropy);
          episode_rewards_vector.push_back(torch::tensor(reward, options));
          // Only check truncated so that bootstrapping is applied for
          // truncated but not for terminated environments.
          if (truncated) {
            add_bootstrap = true;
            break;
          }
          if (terminated) {
            break;
          }
        }

        // Bootstrap value
        if (add_bootstrap) {
          episode_values_vector.push_back(
              agent->get_value(environment.get_observation_as_torch_tensor()));
        } else
          episode_values_vector.push_back(torch::zeros({1}, options));

        torch::Tensor episode_log_probs =
            torch::stack(episode_log_probs_vector);
        torch::Tensor episode_values = torch::stack(episode_values_vector);
        torch::Tensor episode_rewards = torch::stack(episode_rewards_vector);
        torch::Tensor episode_entropies =
            torch::stack(episode_entropies_vector);

        auto [actor_loss, critic_loss] = BaseA3CAgent::get_losses(
            episode_rewards, episode_log_probs, episode_values,
            episode_entropies, discount_factor, gae_hyperparameter,
            entropy_coefficient);

        // In synchronous A2C one would now call agent->update_parameters().
        // However, in A3C we manually compute the gradients, keep them without
        // updating the worker agent, and use them to update the global agent
        // boss.
        actor_loss.backward();
        critic_loss.backward();

        // Truly asynchronous is  Hogwild-style with allowed races.
        // Multiple workes could theoretically call load gradients before the
        // update.
        // Update parameters assuming gradients are loaded uses the inner mutex
        // from agent boss, so no need to apply the global mutex here either.
        agent_boss->load_gradients(*agent);
        agent_boss->update_parameters_assuming_gradients_are_loaded();
        {
          std::lock_guard lock(*global_mutex);
          global_async_step += update_step;
          global_max_reward = std::max(global_max_reward, total_worker_reward);
          updateProgress(
              global_async_step, a3c_max_async_steps,
              " | Reward: " + std::to_string(total_worker_reward) +
                  "Global Max: " + std::to_string(global_max_reward));
        }
      }
    }));
  }
  for (unsigned int i = 0u; i < nr_asynchronous_agents; i++) {
    futures[i].wait();
  }
  if (!interrupted) {
    std::cout << "\nTraining finished." << std::endl;
  }
  return {{"global_max_reward", std::to_string(global_max_reward)}};
}
} // namespace ai_pass_selector