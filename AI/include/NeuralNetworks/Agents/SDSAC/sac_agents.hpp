#ifndef SAC_AGENTS_HPP
#define SAC_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/SAC/base_sac_agent.hpp"

#define DECLARE_SAC_AGENT(ClassName)                                           \
  class ClassName final : public BaseSACAgent {                                \
  public:                                                                      \
    explicit ClassName(unsigned int max_qubits);                               \
    std::string agentName() const override;                                    \
  };

namespace ai_pass_selector {
/// SDSAC = Soft Actor-Critic
/// TCN = Temporal Convolutional Network
/// RELU = Activation Functions are set to ReLU
/// PRELU = Activation Functions are set to PReLU
/// LSTM = Long Short-Term Memory
/// HMPP = High Memory Plus Projection
/// BMNP = Balanced Memory No Projection
/// HYBRID = Hybrid of TCN and LSTM
DECLARE_SAC_AGENT(SAC_TCN_RELU)
DECLARE_SAC_AGENT(SAC_TCN_PRELU)
DECLARE_SAC_AGENT(SAC_LSTM_HMPP)
DECLARE_SAC_AGENT(SAC_LSTM_BMNP)
DECLARE_SAC_AGENT(SAC_HYBRID)
} // namespace ai_pass_selector

#endif // SAC_AGENTS_HPP