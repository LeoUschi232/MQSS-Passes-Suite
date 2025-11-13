#ifndef ACER_TRAINER_HPP
#define ACER_TRAINER_HPP

// Agents includes
#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"

namespace ai_pass_selector {
struct ACER_TrajectoryElement {
  unsigned int action_index;
  torch::Tensor action_probs;
};
struct ACER_Trajectory {
  int environment_reset_seed;
  std::vector<ACER_TrajectoryElement> trajectory_elements;
};

/**
 * @param agent
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train_acer(const std::unique_ptr<BaseACERAgent> &agent,
           const std::string &dataset);
} // namespace ai_pass_selector

#endif // ACER_TRAINER_HPP
