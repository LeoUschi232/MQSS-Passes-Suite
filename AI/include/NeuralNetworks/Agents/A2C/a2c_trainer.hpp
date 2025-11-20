#ifndef A2C_TRAINER_HPP
#define A2C_TRAINER_HPP

#include "base_a2c_agent.hpp"

namespace ai_pass_selector {
/**
 * @param agent
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train_a2c(const std::unique_ptr<BaseA2CAgent> &agent,
          const std::string &dataset);
} // namespace ai_pass_selector

#endif // A2C_TRAINER_HPP
