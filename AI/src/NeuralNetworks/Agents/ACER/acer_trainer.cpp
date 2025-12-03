#include "NeuralNetworks/Agents/ACER/acer_trainer.hpp"

// Environment includes
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
extern torch::TensorOptions GLOBAL_TENSOR_OPTIONS;

std::unordered_map<std::string, std::string>
train_acer(const std::unique_ptr<BaseACERAgent> &agent,
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
      GLOBAL_PARAMS["save_agent_after_training"].to_bool();
  unsigned int acer_max_nr_trajectories =
      GLOBAL_PARAMS["acer_max_nr_trajectories"].to_int();
  unsigned int acer_ratio_of_replay =
      GLOBAL_PARAMS["acer_ratio_of_replay"].to_int();
  torch::Device device = GLOBAL_PARAMS["device"].to_device_type();
  if (nr_episodes <= 0 || max_steps_per_episode <= 0) {
    std::cerr << "Nothing to train." << std::endl;
    return {};
  }
  auto optional_statistics = get_dataset_statistics_from_dataset_name(dataset);
  if (!optional_statistics.has_value()) {
    std::cerr << "Dataset " + dataset + " doesn't have statistics for training."
              << std::endl;
    return {};
  }
  auto [qubits_cholesky_params, gates_weights] = optional_statistics.value();
  QuantumCircuitEnvironment environment(max_qubits);
  environment.register_randomizer_params(qubits_cholesky_params, gates_weights);
  std::cout << "Beginning training." << std::endl;
  updateProgress(0, nr_episodes, "Beginning training");
  //////////////////////////////////////////////////////////////////////////////
  /// Algorithm 1 ACER for discrete actions (master algorithm)
  std::vector<ACER_Trajectory> replay_buffer;
  replay_buffer.reserve(acer_max_nr_trajectories);
  unsigned int off_policy_episodes_left = 0u;
  for (unsigned int episode_idx = 1; episode_idx <= nr_episodes;
       episode_idx++) {
    if (interrupted) {
      break;
    }
    if (episode_idx % save_agent_every_ith_episode == 0) {
      agent->save_model();
      updateProgress(episode_idx, nr_episodes, "Saving Agent.");
    }
    std::string reset_string = "Resetting Enviornment.";
    bool on_policy = true;
    if (off_policy_episodes_left <= 0u) {
      off_policy_episodes_left = randomPoisson(acer_ratio_of_replay);
      if (off_policy_episodes_left > 2 * acer_ratio_of_replay) {
        reset_string += " | Warning: off_policy_episodes_left=" +
                        std::to_string(off_policy_episodes_left);
      }
    } else {
      on_policy = false;
      off_policy_episodes_left--;
    }
    updateProgress(episode_idx, nr_episodes, reset_string);
    int environment_seed = std::random_device{}();
    std::vector<ACER_TrajectoryElement> trajectory_elements;
    if (!on_policy && replay_buffer.empty()) {
      off_policy_episodes_left = 0u;
      on_policy = true;
    }
    //////////////////////////////////////////////////////////////////////////////
    /// Algorithm 2 ACER for discrete actions
    agent->reset_gradients();
    if (!on_policy) {
      unsigned int replay_index = randomInt(0u, replay_buffer.size());
      auto [seed, elements] = replay_buffer[replay_index];
      environment_seed = seed;
      trajectory_elements = elements;
    }
    environment.reset(environment_seed);
    try {
      double total_episode_reward = 0.0;
      auto [nr_qubits, nr_gates] = environment.size();
      std::vector<torch::Tensor> original_policies;
      std::vector<torch::Tensor> policies_main;
      std::vector<torch::Tensor> policies_avg;
      std::vector<torch::Tensor> Q_values_list;
      std::vector<torch::Tensor> rewards;
      std::vector<unsigned int> action_indices;
      original_policies.reserve(max_steps_per_episode);
      policies_main.reserve(max_steps_per_episode);
      policies_avg.reserve(max_steps_per_episode);
      Q_values_list.reserve(max_steps_per_episode);
      rewards.reserve(max_steps_per_episode);
      action_indices.reserve(max_steps_per_episode);

      bool add_bootstrap = false;
      for (unsigned int step_idx = 0u; step_idx < max_steps_per_episode;
           step_idx++) {
        if (interrupted) {
          break;
        }
        updateProgresses(
            {{episode_idx, nr_episodes}, {step_idx + 1, max_steps_per_episode}},
            "Reward: " + std::to_string(total_episode_reward) +
                " | Nr qubits: " + std::to_string(nr_qubits) +
                " | Nr gates: " + std::to_string(nr_gates) + " | Stepping.");
        torch::Tensor observation =
            environment.get_observation_as_torch_tensor();
        auto [policy_main, policy_avg, Q_values] = agent->forward(observation);
        unsigned int action_index;
        if (on_policy) {
          torch::Tensor action = agent->select_action(policy_main);
          action_index = action.item<int>();
          assert(trajectory_elements.size() == step_idx);
          trajectory_elements.push_back({action_index, policy_main.detach()});
        } else {
          action_index = trajectory_elements[step_idx].action_index;
        }
        auto [reward, terminated, truncated] = environment.step(action_index);
        original_policies.push_back(trajectory_elements[step_idx].action_probs);
        policies_main.push_back(policy_main);
        policies_avg.push_back(policy_avg.detach());
        Q_values_list.push_back(Q_values);
        rewards.push_back(torch::tensor(reward, GLOBAL_TENSOR_OPTIONS));
        action_indices.push_back(action_index);
        total_episode_reward += reward;
        if (truncated) {
          add_bootstrap = true;
          break;
        }
        if (terminated) {
          break;
        }
      }
      if (original_policies.empty()) {
        continue;
      }
      torch::Tensor Q_ret = torch::zeros({}, GLOBAL_TENSOR_OPTIONS);
      if (add_bootstrap) {
        torch::NoGradGuard _;
        torch::Tensor observation =
            environment.get_observation_as_torch_tensor();
        Q_ret = agent->get_value_main(observation);
      }
      torch::Tensor final_critic_loss;
      torch::Tensor final_actor_gradients;
      bool first_iteration = true;
      for (int i = static_cast<int>(rewards.size()) - 1; i >= 0; i--) {
        updateProgress(episode_idx, nr_episodes,
                       std::to_string(i) + "->0 | Reward: " +
                           std::to_string(total_episode_reward) +
                           " | Nr qubits: " + std::to_string(nr_qubits) +
                           " | Nr gates: " + std::to_string(nr_gates) +
                           " | Computing losses.");
        auto [actor_gradients, critic_loss, new_Q_ret] =
            agent->compute_losses_and_accumulate_gradients(
                /*reward=*/rewards[i].to(device),
                /*Q_ret=*/Q_ret.to(device),
                /*policy_main=*/policies_main[i].to(device),
                /*policy_avg=*/policies_avg[i].detach().to(device),
                /*Q_values=*/Q_values_list[i].to(device),
                /*original_policy=*/
                original_policies[i].to(device),
                /*action_indices=*/action_indices[i]);
        Q_ret = new_Q_ret.detach();
        if (first_iteration) {
          final_actor_gradients = actor_gradients;
          final_critic_loss = critic_loss;
          first_iteration = false;
        } else {
          final_actor_gradients += actor_gradients;
          final_critic_loss += critic_loss;
        }
      }
      updateProgress(episode_idx, nr_episodes,
                     "Reward: " + std::to_string(total_episode_reward) +
                         " | Nr qubits: " + std::to_string(nr_qubits) +
                         " | Nr gates: " + std::to_string(nr_gates) +
                         " | Updating parameters.");
      agent->update_parameters(final_actor_gradients, final_critic_loss);
      if (on_policy) {
        if (replay_buffer.size() >= acer_max_nr_trajectories) {
          unsigned int remove_index = randomInt(0u, replay_buffer.size());
          replay_buffer.erase(replay_buffer.begin() + remove_index);
        }
        replay_buffer.push_back(
            {environment_seed, std::move(trajectory_elements)});
      }
    } catch (const std::exception &error) {
      std::cerr << "Error in Episode " << episode_idx << ":\n"
                << error.what() << std::endl;
      agent->load_model();
      if (stop_training_on_error) {
        interrupted = 1;
        break;
      }
    }
    //////////////////////////////////////////////////////////////////////////////
  }
  //////////////////////////////////////////////////////////////////////////////
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
