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
  torch::TensorOptions options =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  int64_t T = max_steps_per_episode;
  std::cout << "Beginning training." << std::endl;
  updateProgress(0, nr_episodes, /*display_message=*/"Beginning training");
  //////////////////////////////////////////////////////////////////////////////
  /// TODO: Train ACER Agent
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
    bool on_policy = true;
    if (off_policy_episodes_left <= 0u) {
      off_policy_episodes_left = randomPoisson(acer_ratio_of_replay);
    } else {
      on_policy = false;
      off_policy_episodes_left--;
    }
    updateProgress(/*current=*/episode_idx, /*total=*/nr_episodes,
                   /*display_message=*/"Resetting Enviornment.");






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
