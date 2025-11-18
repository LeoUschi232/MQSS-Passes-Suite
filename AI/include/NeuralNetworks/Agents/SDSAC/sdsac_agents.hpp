#ifndef SDSAC_AGENTS_HPP
#define SDSAC_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/SDSAC/base_sdsac_agent.hpp"

#define DECLARE_SDSAC_AGENT(ClassName)                                         \
  class ClassName final : public BaseSDSACAgent {                              \
  public:                                                                      \
    explicit ClassName(unsigned int max_qubits);                               \
    std::string agentName() const override;                                    \
  };

namespace ai_pass_selector {
/// SDSAC = Stable Discrete Soft Actor-Critic
/// TCN = Temporal Convolutional Network
/// LSTM = Long Short-Term Memory
/// HYBRID = Hybrid of TCN and LSTM
DECLARE_SDSAC_AGENT(SDSAC_TCN)
DECLARE_SDSAC_AGENT(SDSAC_LSTM)
DECLARE_SDSAC_AGENT(SDSAC_HYBRID)
} // namespace ai_pass_selector

#endif // SDSAC_AGENTS_HPP