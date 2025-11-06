#ifndef SAC_TRAINER_HPP
#define SAC_TRAINER_HPP

// Agents includes
#include "NeuralNetworks/Agents/SAC/base_sac_agent.hpp"

namespace ai_pass_selector {

/**
 *
 * @param agent
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train_sac(const std::unique_ptr<BaseSACAgent> &agent,
          const std::string &dataset);
} // namespace ai_pass_selector

#endif // SAC_TRAINER_HPP