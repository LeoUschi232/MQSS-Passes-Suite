#ifndef A2C_AGENTS_HPP
#define A2C_AGENTS_HPP

#include "Torch/A2C/base_a2c_agent.hpp"

#define DECLARE_A2C_AGENT(ClassName)                                           \
  class ClassName final : public ai_pass_selector::BaseA2CAgent {              \
  public:                                                                      \
    ClassName(int circuit_size_class,                                          \
              std::unordered_map<std::string, std::string> params);            \
    ClassName(const std::string &circuit_size,                                 \
              std::unordered_map<std::string, std::string> params);            \
    std::string agentName() const override;                                    \
  };

namespace ai_pass_selector {
/// A2C = Advantage Actor-Critic
/// IB = Instruction-Based
/// DB = Depth-Based
/// FC = Fully Connected
/// CONV = Convolutional
/// LSM = Layer Size Maintaining
/// LSD = Layer Size Decreasing
DECLARE_A2C_AGENT(A2C_IB_FC_LSM)
DECLARE_A2C_AGENT(A2C_IB_FC_LSD)
DECLARE_A2C_AGENT(A2C_IB_CONV_LSM)
DECLARE_A2C_AGENT(A2C_IB_CONV_LSD)
DECLARE_A2C_AGENT(A2C_DB_FC_LSM)
DECLARE_A2C_AGENT(A2C_DB_FC_LSD)
DECLARE_A2C_AGENT(A2C_DB_CONV_LSM)
DECLARE_A2C_AGENT(A2C_DB_CONV_LSD)
} // namespace ai_pass_selector
#endif // A2C_TRAINER_HPP
