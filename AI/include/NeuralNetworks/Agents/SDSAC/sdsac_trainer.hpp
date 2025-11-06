#ifndef SDSAC_TRAINER_HPP
#define SDSAC_TRAINER_HPP

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
train_sdsac(const std::unique_ptr<BaseSACAgent> &agent,
          const std::string &dataset);
} // namespace ai_pass_selector

#endif // SDSAC_TRAINER_HPP