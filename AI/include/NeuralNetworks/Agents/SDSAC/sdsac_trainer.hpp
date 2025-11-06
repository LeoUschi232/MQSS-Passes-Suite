#ifndef SDSAC_TRAINER_HPP
#define SDSAC_TRAINER_HPP

// Agents includes
#include "NeuralNetworks/Agents/SDSAC/base_sdsac_agent.hpp"

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
struct SDSAC_EpisodeRollout {
  std::vector<torch::Tensor> observations; // [T+1, N_t, IRS]
  torch::Tensor actions;                   // [T]
  torch::Tensor rewards;                   // [T]
  torch::Tensor entropies;                 // [T]
};

/**
 *
 * @param agent
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train_sdsac(const std::unique_ptr<BaseSDSACAgent> &agent,
            const std::string &dataset);
} // namespace ai_pass_selector

#endif // SDSAC_TRAINER_HPP