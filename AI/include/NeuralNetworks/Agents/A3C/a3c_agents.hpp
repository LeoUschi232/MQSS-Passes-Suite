#ifndef A3C_AGENTS_HPP
#define A3C_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/A3C/base_a3c_agent.hpp"

#define DECLARE_A3C_AGENT(ClassName)                                           \
  class ClassName final : public BaseA3CAgent {                                \
  public:                                                                      \
    explicit ClassName(unsigned int max_qubits, bool is_boss = true);          \
    std::string agentName() const override;                                    \
    std::unique_ptr<BaseA3CAgent> clone() const override;                      \
  };

namespace ai_pass_selector {
/// A2C = Advantage Actor-Critic
/// A3C = Asynchronous Advantage Actor-Critic
/// TCN = Temporal Convolutional Network
/// RELU = Activation Functions are set to ReLU
/// PRELU = Activation Functions are set to PReLU
/// LSTM = Long Short-Term Memory
/// HMPP = High Memory Plus Projection
/// BMNP = Balanced Memory No Projection
/// HYBRID = Hybrid of TCN and LSTM
// a3c-mq28-tcnrelu Nr trainable parameters: 934434
// a3c-mq130-tcnrelu Nr trainable parameters: 10845966
DECLARE_A3C_AGENT(A3C_TCN_RELU)
// a3c-mq28-tcnprelu Nr trainable parameters: 934494
// a3c-mq130-tcnprelu Nr trainable parameters: 10846038
DECLARE_A3C_AGENT(A3C_TCN_PRELU)
// a3c-mq28-lstmhmpp Nr trainable parameters: 1057986
// a3c-mq130-lstmhmpp Nr trainable parameters: 10160466
DECLARE_A3C_AGENT(A3C_LSTM_HMPP)
// a3c-mq28-lstmbmnp Nr trainable parameters: 1147266
// a3c-mq130-lstmbmnp Nr trainable parameters: 10929066
DECLARE_A3C_AGENT(A3C_LSTM_BMNP)
// a3c-mq28-hybrid Nr trainable parameters: 860994
// a3c-mq130-hybrid Nr trainable parameters: 9284466
DECLARE_A3C_AGENT(A3C_HYBRID)
} // namespace ai_pass_selector
#endif // A3C_AGENTS_HPP
