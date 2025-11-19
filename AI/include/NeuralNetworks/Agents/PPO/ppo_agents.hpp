#ifndef PPO_AGENTS_HPP
#define PPO_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/PPO/base_ppo_agent.hpp"

#define DECLARE_PPO_AGENT(ClassName)                                           \
  class ClassName final : public BasePPOAgent {                                \
  public:                                                                      \
    explicit ClassName(unsigned int max_qubits);                               \
    std::string agentName() const override;                                    \
  };

namespace ai_pass_selector {
/// PPO = Proximal Policy Optimization
/// TCN = Temporal Convolutional Network
/// LSTM = Long Short-Term Memory
/// HYBRID = Hybrid of TCN and LSTM
DECLARE_PPO_AGENT(PPO_TCN)
DECLARE_PPO_AGENT(PPO_LSTM)
DECLARE_PPO_AGENT(PPO_HYBRID)
} // namespace ai_pass_selector

#endif // PPO_AGENTS_HPP
