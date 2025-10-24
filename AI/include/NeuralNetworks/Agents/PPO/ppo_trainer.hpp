#ifndef PPO_TRAINER_HPP
#define PPO_TRAINER_HPP

// Agents includes
#include "NeuralNetworks/Agents/PPO/base_ppo_agent.hpp"

namespace ai_pass_selector {

struct EpisodeRollout {
  std::vector<torch::Tensor> observations; // [N_t, IRS], length T+1
  torch::Tensor actions;                   // [T]
  torch::Tensor log_action_probs;          // [T]
  torch::Tensor state_values;              // [T+1]
  torch::Tensor rewards;                   // [T]
};

/**
 *
 * @param agent
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train_ppo(const std::unique_ptr<BasePPOAgent> &agent,
          const std::string &dataset);

} // namespace ai_pass_selector

#endif // PPO_TRAINER_HPP