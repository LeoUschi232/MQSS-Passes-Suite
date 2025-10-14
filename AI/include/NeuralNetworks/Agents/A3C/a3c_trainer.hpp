#ifndef A3C_TRAINER_HPP
#define A3C_TRAINER_HPP

#include "base_a3c_agent.hpp"

namespace ai_pass_selector {
/**
 *
 * @param agent_boss
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train_a3c(const std::unique_ptr<BaseA3CAgent> &agent_boss,
          const std::string &dataset);

/**
 *
 * @param agent
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train_a2c(const std::unique_ptr<BaseA3CAgent> &agent,
          const std::string &dataset);
} // namespace ai_pass_selector

#endif // A3C_TRAINER_HPP
