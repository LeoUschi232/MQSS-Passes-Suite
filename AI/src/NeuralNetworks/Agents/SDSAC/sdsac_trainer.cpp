#include "NeuralNetworks/Agents/SDSAC/sdsac_trainer.hpp"

// Environment includes
#include "Environment/Wrappers/normalize_reward.hpp"

// Utils includes
#include "Utils/info_utils.hpp"
#include "Utils/progress_bar.hpp"

// Standard library includes
#include <csignal>
#include <filesystem>

namespace fs = std::filesystem;

extern sig_atomic_t interrupted;
extern void signal_handler(int signal);

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

std::unordered_map<std::string, std::string>
train_sdsac(const std::unique_ptr<BaseSDSACAgent> &agent,
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
  SDSAC_EpisodeRollout rollout_old;
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

      SDSAC_EpisodeRollout rollout_new;
      std::vector<torch::Tensor> actions_vector;
      std::vector<torch::Tensor> rewards_vector;
      std::vector<torch::Tensor> entropies_vector;
      rollout_new.observations.reserve(T + 1);
      actions_vector.reserve(T);
      rewards_vector.reserve(T);
      entropies_vector.reserve(T);

      unsigned int update_step;
      for (update_step = 0u; update_step < max_steps_per_episode;
           update_step++) {
        updateProgresses({{episode_idx, nr_episodes},
                          {update_step + 1, max_steps_per_episode}},
                         /*display_message=*/"Rollout A | Reward: " +
                             std::to_string(total_episode_reward) +
                             " | Nr qubits: " + std::to_string(nr_qubits) +
                             " | Nr gates: " + std::to_string(nr_gates));
        if (interrupted) {
          std::cout << "Caught Ctrl+C Interruption in SDSAC training."
                    << std::endl;
          break;
        }
        // TODO: do loop A
        torch::Tensor observation =
            environment.get_observation_as_torch_tensor();
        auto [action, entropy] = agent->sdsac_select_action(observation);
        rollout_new.observations.push_back(observation);
        actions_vector.push_back(action);
        entropies_vector.push_back(entropy);
        auto [reward, terminated, truncated] =
            environment.step(action.item<int>());
        rewards_vector.push_back(torch::tensor(reward, options));
        total_episode_reward += reward;
        if (truncated || terminated) {
          break;
        }
      }
      torch::Tensor observation = environment.get_observation_as_torch_tensor();
      rollout_new.observations.push_back(observation);
      rollout_new.actions = torch::stack(actions_vector).to(device);
      rollout_new.rewards = torch::stack(rewards_vector).to(device);
      rollout_new.entropies = torch::stack(entropies_vector).to(device);

      //////////////////////////////////////////////////////////////////////////
      /// Inner loop 2: Rollout B
      // Observations is length T+1.
      unsigned int steps_in_episode = rollout_old.observations.size();
      // Must reduce to T to match actions, log_action_probs, and rewards.
      if (steps_in_episode-- <= 1u) {
        // Empty rollout, probably first episode, skip update.
        rollout_old = std::move(rollout_new);
        previous_episode_reward = total_episode_reward;
        previous_nr_qubits = nr_qubits;
        previous_nr_gates = nr_gates;
        continue;
      }
      for (update_step = 0u; update_step < steps_in_episode; update_step++) {
        updateProgresses(
            {{episode_idx, nr_episodes}, {update_step + 1, steps_in_episode}},
            /*display_message=*/"Rollout B | Reward: " +
                std::to_string(previous_episode_reward) +
                " | Nr qubits: " + std::to_string(previous_nr_qubits) +
                " | Nr gates: " + std::to_string(previous_nr_gates));
        if (interrupted) {
          std::cout << "Caught Ctrl+C Interruption in SDSAC training."
                    << std::endl;
          break;
        }
        auto [action_probs, Q1_main, Q2_main, Q1_avg, Q2_avg] =
            agent->sdsac_all_Q_forward(
                /*observation=*/rollout_old.observations[update_step]);
        auto [action_probs_next, Q1_avg_next, Q2_avg_next] =
            agent->sdsac_Q_avg_only_forward(
                /*observation=*/rollout_old.observations[update_step + 1u]);

        auto [actor_loss, critic_Q1_loss, critic_Q2_loss,
              optional_temperature_alpha_loss] =
            agent->get_loss(
                /*reward=*/rollout_old.rewards[update_step],
                /*action_probs_next=*/action_probs_next.detach(),
                /*Q1_avg_next=*/Q1_avg_next.detach(),
                /*Q2_avg_next=*/Q2_avg_next.detach(),
                /*Q1_main=*/Q1_main,
                /*Q2_main=*/Q2_main,
                /*Q1_avg=*/Q1_avg.detach(),
                /*Q2_avg=*/Q2_avg.detach(),
                /*action=*/rollout_old.actions[update_step],
                /*action_probs=*/action_probs,
                /*old_entropy=*/rollout_old.entropies[update_step].detach(),
                /*new_entropy=*/
                -(action_probs * action_probs.log()).sum(-1).squeeze(-1));
        agent->sdsac_update_parameters(actor_loss, critic_Q1_loss,
                                       critic_Q2_loss,
                                       optional_temperature_alpha_loss);
      }
      rollout_old = std::move(rollout_new);
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

  return {};
}

} // namespace ai_pass_selector
