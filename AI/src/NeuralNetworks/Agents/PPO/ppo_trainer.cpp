#include "NeuralNetworks/Agents/PPO/ppo_trainer.hpp"

// Environment includes
#include "Environment/Wrappers/normalize_reward.hpp"
#include "Environment/quantum_circuit_environment.hpp"
#include "Environment/statistics_for_rqcg.hpp"

// Utils includes
#include "Utils/info_utils.hpp"
#include "Utils/progress_bar.hpp"

// Standard library includes
#include <csignal>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

extern sig_atomic_t interrupted;
extern void signal_handler(int signal);

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;
std::unordered_map<std::string, std::string>
train_ppo(const std::unique_ptr<BasePPOAgent> &agent,
          const std::string &dataset) {
  // Default values
  unsigned int max_qubits = agent->getMaxQubits();
  unsigned int nr_episodes = GLOBAL_PARAMS["nr_episodes"].to_int();
  unsigned int max_steps_per_episode =
      GLOBAL_PARAMS["max_steps_per_episode"].to_int();
  unsigned int save_agent_every_ith_episode =
      GLOBAL_PARAMS["save_agent_every_ith_episode"].to_int();
  torch::Device device = GLOBAL_PARAMS["device"].to_device_type();
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
  torch::TensorOptions options =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  int64_t T = max_steps_per_episode;
  std::cout << "Beginning training." << std::endl;

  updateProgress(0, nr_episodes, /*display_message=*/"Beginning training");
  EpisodeRollout rollout_old;
  bool add_bootstrap_old = false;
  double previous_episode_reward = 0.0;
  unsigned int previous_nr_qubits = 0u;
  unsigned int previous_nr_gates = 0u;
  for (unsigned int episode_idx = 1; episode_idx <= nr_episodes;
       episode_idx++) {
    if (interrupted) {
      break;
    }
    if (episode_idx % save_agent_every_ith_episode == 0) {
      agent->save_model();
      updateProgress(/*current=*/episode_idx, /*total=*/nr_episodes,
                     /*display_message=*/"Saving Agent.");
    }
    updateProgress(/*current=*/episode_idx, /*total=*/nr_episodes,
                   /*display_message=*/"Resetting Enviornment.");
    try {
      //////////////////////////////////////////////////////////////////////////
      /// Inner loop 1: Rollout A
      double total_episode_reward = 0.0;
      environment.reset();
      auto [nr_qubits, nr_gates] = environment.size();

      EpisodeRollout rollout_new;
      std::vector<torch::Tensor> actions_vector;
      std::vector<torch::Tensor> log_action_probs_vector;
      std::vector<torch::Tensor> state_values_vector;
      std::vector<torch::Tensor> rewards_vector;
      rollout_new.observations.reserve(T + 1);
      actions_vector.reserve(T);
      log_action_probs_vector.reserve(T);
      state_values_vector.reserve(T + 1);
      rewards_vector.reserve(T);

      bool add_bootstrap_new = false;
      unsigned int update_step;
      for (update_step = 0u; update_step < max_steps_per_episode;
           update_step++) {
        updateProgresses({{episode_idx, nr_episodes},
                          {update_step + 1, max_steps_per_episode}},
                         /*display_message=*/"Rollout A | Episode Reward: " +
                             std::to_string(total_episode_reward) +
                             " | Nr qubits: " + std::to_string(nr_qubits) +
                             " | Nr gates: " + std::to_string(nr_gates));
        if (interrupted) {
          std::cout << "Caught Ctrl+C Interruption in PPO training."
                    << std::endl;
          break;
        }

        torch::Tensor observation =
            environment.get_observation_as_torch_tensor();
        auto [action, log_action_prob, state_value, _] =
            agent->select_action(observation);
        rollout_new.observations.push_back(observation);
        actions_vector.push_back(action);
        log_action_probs_vector.push_back(log_action_prob);
        state_values_vector.push_back(state_value);
        auto [reward, terminated, truncated] =
            environment.step(action.item<int>());
        rewards_vector.push_back(torch::tensor(reward, options));
        total_episode_reward += reward;
        if (truncated) {
          add_bootstrap_new = true;
          break;
        }
        if (terminated) {
          break;
        }
      }
      torch::Tensor observation = environment.get_observation_as_torch_tensor();
      rollout_new.observations.push_back(observation);
      if (add_bootstrap_new || update_step >= max_steps_per_episode) {
        torch::NoGradGuard _;
        state_values_vector.push_back(agent->get_value(observation));
      } else {
        state_values_vector.push_back(torch::zeros({}, options));
      }
      rollout_new.actions = torch::stack(actions_vector).to(device);
      rollout_new.log_action_probs =
          torch::stack(log_action_probs_vector).to(device);
      rollout_new.state_values = torch::stack(state_values_vector).to(device);
      rollout_new.rewards = torch::stack(rewards_vector).to(device);
      //////////////////////////////////////////////////////////////////////////
      /// Inner loop 2: Rollout B
      // Observations is length T+1.
      unsigned int steps_in_episode = rollout_old.observations.size();
      // Must reduce to T to match actions, log_action_probs, and rewards.
      if (steps_in_episode-- <= 1u) {
        // Empty rollout, probably first episode, skip update.
        rollout_old = std::move(rollout_new);
        add_bootstrap_old = add_bootstrap_new;
        continue;
      }
      log_action_probs_vector.clear();
      state_values_vector.clear();
      std::vector<torch::Tensor> entropies_vector;
      for (update_step = 0u; update_step < steps_in_episode; update_step++) {
        updateProgresses(
            {{episode_idx, nr_episodes}, {update_step + 1, steps_in_episode}},
            /*display_message=*/"Rollout B | Episode Reward: " +
                std::to_string(previous_episode_reward) +
                " | Nr qubits: " + std::to_string(previous_nr_qubits) +
                " | Nr gates: " + std::to_string(previous_nr_gates));
        if (interrupted) {
          std::cout << "Caught Ctrl+C Interruption in PPO training."
                    << std::endl;
          break;
        }
        auto [log_action_probs, state_value, entropy] =
            agent->force_select_action(
                /*observation=*/rollout_old.observations[update_step],
                /*action_index_unsqueezed=*/rollout_old.actions[update_step]
                    .unsqueeze(-1));
        log_action_probs_vector.push_back(log_action_probs);
        state_values_vector.push_back(state_value);
        entropies_vector.push_back(entropy);
      }
      if (add_bootstrap_old) {
        state_values_vector.push_back(
            agent->get_value(rollout_old.observations.back()));
      } else {
        state_values_vector.push_back(torch::zeros({}, options));
      }

      //////////////////////////////////////////////////////////////////////////
      /// Compute Losses
      std::string main_message =
          "Episode Reward: " + std::to_string(total_episode_reward) +
          " | Nr qubits: " + std::to_string(nr_qubits) +
          " | Nr gates: " + std::to_string(nr_gates);
      updateProgress(/*current=*/episode_idx, /*total=*/nr_episodes,
                     /*display_message=*/main_message + " | Computing loss.");
      auto [actor_loss, critic_loss] = agent->get_losses(
          /*old_log_action_probs=*/rollout_old.log_action_probs.detach().to(
              device),
          /*old_state_values=*/
          rollout_old.state_values.detach().to(device),
          /*new_log_action_probs=*/
          torch::stack(log_action_probs_vector).to(device),
          /*new_state_values=*/
          torch::stack(state_values_vector).to(device),
          /*rewards=*/rollout_old.rewards.to(device),
          /*entropy=*/torch::stack(entropies_vector).to(device));

      //////////////////////////////////////////////////////////////////////////
      /// Update Parameters
      updateProgress(/*current=*/episode_idx, /*total=*/nr_episodes,
                     /*display_message=*/main_message + " | Updating params.");
      agent->update_parameters(actor_loss, critic_loss);
      rollout_old = std::move(rollout_new);
      add_bootstrap_old = add_bootstrap_new;
      previous_episode_reward = total_episode_reward;
      previous_nr_qubits = nr_qubits;
      previous_nr_gates = nr_gates;
    } catch (const std::exception &error) {
      std::cerr << "Episode " << episode_idx << ": " << error.what()
                << std::endl;
      if (GLOBAL_PARAMS["stop_training_on_error"].to_bool()) {
        interrupted = 1;
        break;
      }
    }
  }
  if (!interrupted) {
    std::cout << "\nTraining finished." << std::endl;
  }
  if (GLOBAL_PARAMS["save_agent_after_training"].to_bool()) {
    agent->save_model();
    std::cout << "Saved: " << agent->agentName() << std::endl;
  }
  return {};
}
} // namespace ai_pass_selector