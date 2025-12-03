#ifndef A2C_AGENTS_HPP
#define A2C_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/A2C/base_a2c_agent.hpp"

#define DECLARE_A2C_AGENT(ClassName)                                           \
  class ClassName final : public BaseA2CAgent {                                \
  public:                                                                      \
    explicit ClassName(unsigned int max_qubits);                               \
    std::string agentName() const override;                                    \
  };

namespace ai_pass_selector {
/// A2C = Advantage Actor-Critic
/// A2C = Asynchronous Advantage Actor-Critic
/// TCN = Temporal Convolutional Network
/// LSTM = Long Short-Term Memory
/// HYBRID = Hybrid of TCN and LSTM
// a2c-mq28-tcn Nr trainable parameters:
// a2c-mq130-tcn Nr trainable parameters:
DECLARE_A2C_AGENT(A2C_TCN)
// a2c-mq28-lstm Nr trainable parameters:
// a2c-mq130-lstm Nr trainable parameters:
DECLARE_A2C_AGENT(A2C_LSTM)
// a2c-mq28-hybrid Nr trainable parameters:
// a2c-mq130-hybrid Nr trainable parameters:
DECLARE_A2C_AGENT(A2C_HYBRID)
} // namespace ai_pass_selector
#endif // A2C_AGENTS_HPP
