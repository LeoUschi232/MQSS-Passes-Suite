#ifndef A3C_AGENTS_HPP
#define A3C_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/A3C/base_a3c_agent.hpp"

#define DECLARE_A3C_AGENT(ClassName)                                           \
  class ClassName final : public ai_pass_selector::BaseA3CAgent {              \
  public:                                                                      \
    ClassName(unsigned int max_qubits, bool is_boss = true);                   \
    std::string agentName() const override;                                    \
    std::unique_ptr<BaseA3CAgent> clone() const override;                      \
  };

namespace ai_pass_selector {
/// A2C = Advantage Actor-Critic
/// A3C = Asynchronous Advantage Actor-Critic
/// TCN = Temporal Convolutional Network
/// LSTM = Long Short-Term Memory
/// RELU = Activation Functions are set to ReLU
/// PRELU = Activation Functions are set to PReLU
DECLARE_A3C_AGENT(A3C_TCN_RELU)
DECLARE_A3C_AGENT(A3C_TCN_PRELU)
DECLARE_A3C_AGENT(A3C_LSTM_RELU)
DECLARE_A3C_AGENT(A3C_LSTM_PRELU)
} // namespace ai_pass_selector
#endif // A3C_TRAINER_HPP
