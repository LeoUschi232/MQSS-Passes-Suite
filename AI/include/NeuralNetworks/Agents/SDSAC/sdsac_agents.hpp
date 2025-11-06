#ifndef SDSAC_AGENTS_HPP
#define SDSAC_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/SAC/base_sac_agent.hpp"

#define DECLARE_SDSAC_AGENT(ClassName)                                           \
  class ClassName final : public BaseSDSACAgent {                                \
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
DECLARE_SDSAC_AGENT(SDSAC_TCN_RELU)
DECLARE_SDSAC_AGENT(SDSAC_TCN_PRELU)
DECLARE_SDSAC_AGENT(SDSAC_LSTM_HMPP)
DECLARE_SDSAC_AGENT(SDSAC_LSTM_BMNP)
DECLARE_SDSAC_AGENT(SDSAC_HYBRID)
} // namespace ai_pass_selector

#endif // SDSAC_AGENTS_HPP