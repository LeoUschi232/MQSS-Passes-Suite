#ifndef ACER_TRAINER_HPP
#define ACER_TRAINER_HPP

// Agents includes
#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"

namespace ai_pass_selector {

struct ACER_EpisodeRollout {
  std::vector<torch::Tensor> observations; // [T+1, N_t, IRS]
  torch::Tensor actions;                   // [T]
  torch::Tensor log_action_probs;          // [T]
  torch::Tensor rewards;                   // [T]
};

/**
 *
 * @param agent
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train_acer(const std::unique_ptr<BaseACERAgent> &agent,
           const std::string &dataset);
} // namespace ai_pass_selector

#endif // ACER_TRAINER_HPP
