#include "NeuralNetworks/Agents/A2C/a2c_trainer.hpp"

// Agents includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Environment includes
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
train_a2c(const std::unique_ptr<BaseA2CAgent> &agent,
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
  QuantumCircuitEnvironment environment(max_qubits);
  environment.register_randomizer_params(qubits_cholesky_params, gates_weights);
  int64_t T = max_steps_per_episode;

  std::cout << "Beginning training." << std::endl;
  updateProgress(0, nr_episodes, "Beginning training");
  for (unsigned int episode_idx = 1; episode_idx <= nr_episodes;
       episode_idx++) {
    if (interrupted) {
      std::cout << "\nCaught Ctrl+C Interruption in A2C training." << std::endl;
      break;
    }

    if (episode_idx % save_agent_every_ith_episode == 0) {
      agent->save_model();
      updateProgress(episode_idx, nr_episodes, "Saving Agent.");
    }
    updateProgress(episode_idx, nr_episodes, "Resetting Enviornment.");
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
                         "Reward: " + std::to_string(total_episode_reward) +
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
                       main_message + " | Updating params.");
      agent->update_parameters(actor_loss, critic_loss);
    } catch (const std::exception &error) {
      std::cerr << "Error in Episode " << episode_idx << ":\n"
                << error.what() << std::endl;
      agent->load_model();
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
