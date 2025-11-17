#include "NeuralNetworks/Agents/A3C/a3c_trainer.hpp"

// Agents includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Environment includes
#include "Environment/Wrappers/normalize_reward.hpp"
#include "Environment/quantum_circuit_environment.hpp"
#include "Environment/statistics_for_rqcg.hpp"

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

extern sig_atomic_t interrupted;
extern void signal_handler(int signal);

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;
extern torch::TensorOptions GLOBAL_TENSOR_OPTIONS;

std::unordered_map<std::string, std::string>
train_a3c(const std::unique_ptr<BaseA3CAgent> &agent_boss,
          const std::string &dataset) {
  unsigned int max_qubits = agent_boss->getMaxQubits();

  // Default values
  unsigned int nr_asynchronous_agents =
      GLOBAL_PARAMS["nr_asynchronous_agents"].to_int();
  unsigned int a3c_max_async_steps =
      GLOBAL_PARAMS["a3c_max_async_steps"].to_int();
  unsigned int max_steps_per_episode =
      GLOBAL_PARAMS["max_steps_per_episode"].to_int();
  unsigned int save_agent_every_ith_episode =
      GLOBAL_PARAMS["save_agent_every_ith_episode"].to_int();
  bool stop_training_on_error =
      GLOBAL_PARAMS["stop_training_on_error"].to_bool();
  bool save_agent_after_training =
      GLOBAL_PARAMS["save_agent_after_training"].to_bool();

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

  float global_max_reward = -std::numeric_limits<float>::max();
  int64_t T = max_steps_per_episode;

  auto global_mutex = std::make_unique<std::mutex>();
  unsigned int global_async_step = 0u;
  unsigned int global_episode = 0u;
  std::cout << "Beginning training." << std::endl;
  updateProgress(0u, a3c_max_async_steps, "Beginning training");

  std::vector<std::future<void>> futures;
  futures.reserve(nr_asynchronous_agents);
  std::signal(SIGINT, signal_handler);
  for (unsigned int i = 0u; i < nr_asynchronous_agents; i++) {
    futures.emplace_back(std::async(std::launch::async, [&] {
      NormalizeReward environment(QuantumCircuitEnvironment{max_qubits});
      environment.register_randomizer_params(qubits_cholesky_params,
                                             gates_weights);
      std::unique_ptr<BaseA3CAgent> agent = agent_boss->clone();

      // A3C doesn't have a max cap on the nr of qpisodes.
      // The cap is implicit on the nr of total steps performed by all
      // agents at the same time.
      try {
        while (true) {
          {
            std::lock_guard lock(*global_mutex);
            if (global_episode >= save_agent_every_ith_episode) {
              agent_boss->save_model();
              global_episode -= save_agent_every_ith_episode;
            }
            if (global_async_step >= a3c_max_async_steps) {
              break;
            }
          }
          if (interrupted) {
            break;
          }
          environment.reset();
          agent->zero_grad();
          // Load params uses the inner mutex from both agents, so no need to
          // apply the global mutex here.
          agent->load_weights(*agent_boss);

          std::vector<torch::Tensor> episode_log_probs_vector;
          std::vector<torch::Tensor> episode_values_vector;
          std::vector<torch::Tensor> episode_rewards_vector;
          std::vector<torch::Tensor> episode_entropies_vector;
          episode_log_probs_vector.reserve(T);
          episode_values_vector.reserve(T + 1);
          episode_rewards_vector.reserve(T);
          episode_entropies_vector.reserve(T);

          float total_worker_reward = 0.0;
          unsigned int steps_taken = 0u;
          bool add_bootstrap = false;
          for (unsigned int update_step = 0u;
               update_step < max_steps_per_episode; update_step++) {
            if (interrupted) {
              break;
            }

            auto [action, log_action_probs, state_values, step_entropy] =
                agent->select_action(
                    environment.get_observation_as_torch_tensor());
            auto [reward, terminated, truncated] =
                environment.step(action.item<int>());
            total_worker_reward += reward;
            steps_taken++;

            episode_log_probs_vector.push_back(log_action_probs);
            episode_values_vector.push_back(state_values);
            episode_entropies_vector.push_back(step_entropy);
            episode_rewards_vector.push_back(
                torch::tensor(reward, GLOBAL_TENSOR_OPTIONS));
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
          if (interrupted) {
            std::cout << "\nCaught Ctrl+C Interruption in A3C training."
                      << std::endl;
            break;
          }

          // Bootstrap value
          if (add_bootstrap) {
            torch::NoGradGuard _;
            episode_values_vector.push_back(agent->get_value(
                environment.get_observation_as_torch_tensor()));
          } else {
            episode_values_vector.push_back(
                torch::zeros({}, GLOBAL_TENSOR_OPTIONS));
          }
          auto [actor_loss, critic_loss] = agent->get_losses(
              /*log_action_probs=*/torch::stack(episode_log_probs_vector),
              /*state_values=*/torch::stack(episode_values_vector),
              /*rewards=*/torch::stack(episode_rewards_vector),
              /*entropy=*/torch::stack(episode_entropies_vector));

          // In synchronous A2C one would now call agent->update_parameters().
          // However, in A3C we manually compute the gradients, keep them
          // without updating the worker agent, and use them to update the
          // global agent boss.
          actor_loss.backward();
          critic_loss.backward();

          // Truly asynchronous is  Hogwild-style with allowed races.
          // Multiple workes could theoretically call load gradients before the
          // update.
          // Update parameters assuming gradients are loaded uses the inner
          // mutex from agent boss, so no need to apply the global mutex here
          // either.
          agent_boss->load_gradients(*agent);
          agent_boss->update_parameters_assuming_gradients_are_loaded();
          {
            std::lock_guard lock(*global_mutex);
            global_episode++;
            global_async_step += steps_taken;
            global_max_reward =
                std::max(global_max_reward, total_worker_reward);
            updateProgress(
                /*current=*/global_async_step, /*total=*/a3c_max_async_steps,
                /*display_message=*/"Reward: " +
                    std::to_string(total_worker_reward) +
                    "| Global Max: " + std::to_string(global_max_reward));
          }
        }
      } catch (const std::exception &error) {
        // No need to lock global mutex because global_async_step is only
        // accessed for rough diagnostics, it doesn't have to be exact.
        std::cerr << "\nError in Step " << global_async_step << ":\n"
                  << cut_to_newline(error.what()) << std::endl;
        if (stop_training_on_error) {
          interrupted = 1;
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
  if (save_agent_after_training) {
    agent_boss->save_model();
    std::cout << "Saved: " << agent_boss->agentName() << std::endl;
  }
  return {{"global_max_reward", std::to_string(global_max_reward)}};
}

std::unordered_map<std::string, std::string>
train_a2c(const std::unique_ptr<BaseA3CAgent> &agent,
          const std::string &dataset) {
  // Default values
  unsigned int max_qubits = agent->getMaxQubits();
  unsigned int nr_episodes = GLOBAL_PARAMS["nr_episodes"].to_int();
  unsigned int max_steps_per_episode =
      GLOBAL_PARAMS["max_steps_per_episode"].to_int();
  unsigned int save_agent_every_ith_episode =
      GLOBAL_PARAMS["save_agent_every_ith_episode"].to_int();
  bool stop_training_on_error =
      GLOBAL_PARAMS["stop_training_on_error"].to_bool();
  bool save_agent_after_training =
      GLOBAL_PARAMS["save_agent_after_training_"].to_bool();

  if (nr_episodes <= 0 || max_steps_per_episode <= 0) {
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
  NormalizeReward environment(QuantumCircuitEnvironment{max_qubits});
  environment.register_randomizer_params(qubits_cholesky_params, gates_weights);
  int64_t T = max_steps_per_episode;

  std::cout << "Beginning training." << std::endl;
  updateProgress(0, nr_episodes, /*display_message=*/"Beginning training");
  for (unsigned int episode_idx = 1; episode_idx <= nr_episodes;
       episode_idx++) {
    if (interrupted) {
      std::cout << "\nCaught Ctrl+C Interruption in A2C training." << std::endl;
      break;
    }

    if (episode_idx % save_agent_every_ith_episode == 0) {
      agent->save_model();
      updateProgress(episode_idx, nr_episodes,
                     /*display_message=*/"Saving Agent.");
    }
    updateProgress(episode_idx, nr_episodes,
                   /*display_message=*/"Resetting Enviornment.");
    try {
      double total_episode_reward = 0.0;
      environment.reset();
      auto [nr_qubits, nr_gates] = environment.size();
      std::vector<torch::Tensor> episode_log_probs_vector;
      std::vector<torch::Tensor> episode_values_vector;
      std::vector<torch::Tensor> episode_rewards_vector;
      std::vector<torch::Tensor> episode_entropies_vector;
      episode_log_probs_vector.reserve(T);
      episode_values_vector.reserve(T + 1);
      episode_rewards_vector.reserve(T);
      episode_entropies_vector.reserve(T);

      bool add_bootstrap = false;
      for (unsigned int update_step = 0u; update_step < max_steps_per_episode;
           update_step++) {
        updateProgresses({{episode_idx, nr_episodes},
                          {update_step + 1, max_steps_per_episode}},
                         /*display_message=*/"Reward: " +
                             std::to_string(total_episode_reward) +
                             " | Nr qubits: " + std::to_string(nr_qubits) +
                             " | Nr gates: " + std::to_string(nr_gates) +
                             " | Running step.");
        if (interrupted) {
          break;
        }

        auto [action, log_action_probs, state_values, step_entropy] =
            agent->select_action(environment.get_observation_as_torch_tensor());
        auto [reward, terminated, truncated] =
            environment.step(action.item<int>());

        episode_log_probs_vector.push_back(log_action_probs);
        episode_values_vector.push_back(state_values);
        episode_entropies_vector.push_back(step_entropy);
        total_episode_reward += reward;
        episode_rewards_vector.push_back(
            torch::tensor(reward, GLOBAL_TENSOR_OPTIONS));
        if (truncated) {
          add_bootstrap = true;
          break;
        }
        if (terminated) {
          break;
        }
      }
      if (interrupted) {
        std::cout << "\nCaught Ctrl+C Interruption in A2C training."
                  << std::endl;
        break;
      }
      // Bootstrap value
      if (add_bootstrap) {
        torch::NoGradGuard _;
        episode_values_vector.push_back(
            agent->get_value(environment.get_observation_as_torch_tensor()));
      } else {
        episode_values_vector.push_back(
            torch::zeros({}, GLOBAL_TENSOR_OPTIONS));
      }
      std::string main_message =
          "Reward: " + std::to_string(total_episode_reward) +
          " | Nr qubits: " + std::to_string(nr_qubits) +
          " | Nr gates: " + std::to_string(nr_gates);
      updateProgresses({{episode_idx, nr_episodes},
                        {max_steps_per_episode, max_steps_per_episode}},
                       /*display_message=*/main_message + " | Computing loss.");
      auto [actor_loss, critic_loss] = agent->get_losses(
          /*log_action_probs=*/torch::stack(episode_log_probs_vector),
          /*state_values=*/torch::stack(episode_values_vector),
          /*rewards=*/torch::stack(episode_rewards_vector),
          /*entropy=*/torch::stack(episode_entropies_vector));
      updateProgresses({{episode_idx, nr_episodes},
                        {max_steps_per_episode, max_steps_per_episode}},
                       /*display_message=*/main_message +
                           " | Updating params.");
      agent->update_parameters(actor_loss, critic_loss);
    } catch (const std::exception &error) {
      std::cerr << "\nError in Episode " << episode_idx << ":\n"
                << cut_to_newline(error.what()) << std::endl;
      if (stop_training_on_error) {
        interrupted = 1;
        break;
      }
    }
  }
  if (!interrupted) {
    std::cout << "\nTraining finished." << std::endl;
  }
  if (save_agent_after_training) {
    agent->save_model();
    std::cout << "Saved: " << agent->agentName() << std::endl;
  }
  return {};
}
} // namespace ai_pass_selector
