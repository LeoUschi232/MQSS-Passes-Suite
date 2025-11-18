#ifndef ACER_AGENTS_HPP
#define ACER_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"

#define DECLARE_ACER_AGENT(ClassName)                                          \
  class ClassName final : public BaseACERAgent {                               \
  public:                                                                      \
    explicit ClassName(unsigned int max_qubits);                               \
    std::string agentName() const override;                                    \
  };

namespace ai_pass_selector {
/// SDSAC = Stable Discrete Soft Actor-Critic
/// TCN = Temporal Convolutional Network
/// LSTM = Long Short-Term Memory
/// HYBRID = Hybrid of TCN and LSTM
DECLARE_ACER_AGENT(ACER_TCN)
DECLARE_ACER_AGENT(ACER_LSTM)
DECLARE_ACER_AGENT(ACER_HYBRID)
} // namespace ai_pass_selector

#endif // ACER_AGENTS_HPP
