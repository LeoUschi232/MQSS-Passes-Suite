#include "NeuralNetworks/Agents/ACER/acer_trainer.hpp"

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
  auto optional_statistics = get_precomputed_dataset_statistics(dataset);
  if (!optional_statistics.has_value()) {
    std::cerr << "Dataset " + dataset + " doesn't have statistics for training."
              << std::endl;
    return {};
  }
  auto [qubits_cholesky_params, gates_weights] = optional_statistics.value();
  NormalizeReward environment(QuantumCircuitEnvironment{max_qubits});
  environment.register_randomizer_params(qubits_cholesky_params, gates_weights);
  std::cout << "Beginning training." << std::endl;
  updateProgress(0, nr_episodes, /*display_message=*/"Beginning training");
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
      updateProgress(/*current=*/episode_idx, /*total=*/nr_episodes,
                     /*display_message=*/"Saving Agent.");
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
    updateProgress(/*current=*/episode_idx, /*total=*/nr_episodes,
                   /*display_message=*/reset_string);
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
      std::vector<torch::Tensor> truncated_importance_weights;
      truncated_importance_weights.reserve(max_steps_per_episode);
      std::vector<torch::Tensor> policies_main;
      policies_main.reserve(max_steps_per_episode);
      std::vector<torch::Tensor> policies_avg;
      policies_avg.reserve(max_steps_per_episode);
      std::vector<torch::Tensor> Q_values_list;
      Q_values_list.reserve(max_steps_per_episode);
      std::vector<torch::Tensor> rewards;
      rewards.reserve(max_steps_per_episode);
      std::vector<unsigned int> action_indices;
      action_indices.reserve(max_steps_per_episode);
      unsigned int step_idx;
      for (step_idx = 0u; step_idx < max_steps_per_episode; step_idx++) {
        if (interrupted) {
          break;
        }
        updateProgresses(
            {{episode_idx, nr_episodes}, {step_idx + 1, max_steps_per_episode}},
            /*display_message=*/"Reward: " +
                std::to_string(total_episode_reward) +
                " | Nr qubits: " + std::to_string(nr_qubits) + " | Nr gates: " +
                std::to_string(nr_gates) + " | Running step.");
        torch::Tensor observation =
            environment.get_observation_as_torch_tensor();
        auto [policy_main, policy_avg, Q_values] = agent->forward(observation);
        unsigned int action_index;
        if (on_policy) {
          torch::Tensor action = agent->select_action(policy_main);
          action_index = action.item<unsigned int>();
          assert(trajectory_elements.size() == step_idx);
          trajectory_elements.push_back(
              {/*action_index=*/action_index,
               /*action_probs=*/policy_main.detach()});
        } else {
          action_index = trajectory_elements[step_idx].action_index;
        }
        auto [reward, terminated, truncated] =
            environment.step(/*action=*/action_index);
        // One element of truncated_importance_weights has shape [NR_PASSES].
        // This is because we need to sum a value which is dependent on ρi(a)
        // over all values of a.
        truncated_importance_weights.push_back(
            torch::min(torch::tensor(1.0, GLOBAL_TENSOR_OPTIONS),
                       policy_main /
                           (trajectory_elements[step_idx].action_probs +
                            DIVISION_BY_ZERO_BLOCK))
                .detach());
        policies_main.push_back(policy_main);
        policies_avg.push_back(policy_avg.detach());
        Q_values_list.push_back(Q_values);
        rewards.push_back(torch::tensor(reward, GLOBAL_TENSOR_OPTIONS));
        action_indices.push_back(action_index);
        total_episode_reward += reward;
        if (terminated || truncated) {
          break;
        }
      }
      torch::Tensor Q_ret = torch::zeros({}, GLOBAL_TENSOR_OPTIONS);
      if (step_idx < max_steps_per_episode) {
        torch::NoGradGuard _;
        torch::Tensor observation =
            environment.get_observation_as_torch_tensor();
        Q_ret = agent->get_value_main(observation);
      }
      assert(step_idx > 0u);
      agent->compute_losses_and_accumulate_gradients(
          /*k=*/step_idx,
          /*rewards=*/torch::stack(rewards).to(device),
          /*Q_ret=*/Q_ret.to(device),
          /*policies_main=*/torch::stack(policies_main).to(device),
          /*policies_avg=*/torch::stack(policies_avg).detach().to(device),
          /*Q_values_list=*/torch::stack(Q_values_list).to(device),
          /*truncated_importance_weights=*/
          torch::stack(truncated_importance_weights).detach().to(device),
          /*action_indices=*/action_indices);
      agent->update_assuming_gradients_are_computed();
      if (on_policy) {
        if (replay_buffer.size() >= acer_max_nr_trajectories) {
          unsigned int remove_index = randomInt(0u, replay_buffer.size());
          replay_buffer.erase(/*position=*/replay_buffer.begin() +
                              remove_index);
        }
        replay_buffer.push_back(
            {/*environment_reset_seed=*/environment_seed,
             /*trajectory_elements=*/std::move(trajectory_elements)});
      }
    } catch (const std::exception &) {
      std::cerr << "Error in Episode " << episode_idx << "." << std::endl;
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
