#ifndef ACER_AGENTS_HPP
#define ACER_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"

#define DECLARE_ACER_AGENT(ClassName)                                         \
  class ClassName final : public BaseACERAgent {                              \
  public:                                                                      \
    explicit ClassName(unsigned int max_qubits);                               \
    std::string agentName() const override;                                    \
  };

namespace ai_pass_selector {
/// SDSAC = Stable Discrete Soft Actor-Critic
/// TCN = Temporal Convolutional Network
/// RELU = Activation Functions are set to ReLU
/// PRELU = Activation Functions are set to PReLU
/// LSTM = Long Short-Term Memory
/// HMPP = High Memory Plus Projection
/// BMNP = Balanced Memory No Projection
/// HYBRID = Hybrid of TCN and LSTM
DECLARE_ACER_AGENT(ACER_TCN_RELU)
DECLARE_ACER_AGENT(ACER_TCN_PRELU)
DECLARE_ACER_AGENT(ACER_LSTM_HMPP)
DECLARE_ACER_AGENT(ACER_LSTM_BMNP)
DECLARE_ACER_AGENT(ACER_HYBRID)
} // namespace ai_pass_selector

#endif // ACER_AGENTS_HPP
