#ifndef A3C_AGENTS_HPP
#define A3C_AGENTS_HPP

// Torch includes
#include "Agents/A3C/base_a3c_agent.hpp"

#define DECLARE_A3C_AGENT(ClassName)                                           \
  class ClassName final : public ai_pass_selector::BaseA3CAgent {              \
  public:                                                                      \
    ClassName(unsigned int max_qubits,                                         \
              std::unordered_map<std::string, std::string> params,             \
              bool is_boss = true);                                            \
    std::string agentName() const override;                                    \
    std::unique_ptr<BaseA3CAgent> clone() const override;                      \
  };


namespace ai_pass_selector {
/// A2C = Advantage Actor-Critic
/// A3C = Asynchronous Advantage Actor-Critic
/// TCN = Temporal Convolutional Network
DECLARE_A3C_AGENT(A3C_TCN)
} // namespace ai_pass_selector
#endif // A3C_TRAINER_HPP
