#ifndef A3C_AGENTS_HPP
#define A3C_AGENTS_HPP

// Torch includes
#include "NeuralNetworks/Agents/A3C/base_a3c_agent.hpp"

#define DECLARE_A3C_AGENT(ClassName)                                           \
  class ClassName final : public BaseA3CAgent {                                \
  public:                                                                      \
    ClassName(unsigned int max_qubits, bool is_boss = true);                   \
    std::string agentName() const override;                                    \
    std::unique_ptr<BaseA3CAgent> clone() const override;                      \
  };

namespace ai_pass_selector {
// For TCN/Hybrid architectures the depth, and with it the maximum receptive
// field, should be dependant on the maximum number of qubits in the circuits.
// The authors of the paper on TCN recommended depths between 8 and 12.
// It makes sense make the size of the receptive field scale linearly with the
// number of qubits.
// Arbitrarily choosing a depth of 12 for pure TCN architectures for the
// MQTBench case of max_qubits=130, we have to find an 'x' such that:
// log2(x*130)=12 => x=31.5.
constexpr double TCN_QUBIT_MAGIC = 31.5;
// Arbitrarily choosing a depth of 9 for pure Hybrid architectures for the
// MQTBench case of max_qubits=130, we have to find an 'x' such that:
// log2(x*130)=9 => x=3.94.
constexpr double HYBRID_QUBIT_MAGIC = 3.93;

/// A2C = Advantage Actor-Critic
/// A3C = Asynchronous Advantage Actor-Critic
/// TCN = Temporal Convolutional Network
/// RELU = Activation Functions are set to ReLU
/// PRELU = Activation Functions are set to PReLU
/// LSTM = Long Short-Term Memory
/// HMPP = High Memory Plus Projection
/// BMNP = Balanced Memory No Projection
/// HYBRID = Hybrid of TCN and LSTM
DECLARE_A3C_AGENT(A3C_TCN_RELU)
DECLARE_A3C_AGENT(A3C_TCN_PRELU)
DECLARE_A3C_AGENT(A3C_LSTM_HMPP)
DECLARE_A3C_AGENT(A3C_LSTM_BMNP)
DECLARE_A3C_AGENT(A3C_HYBRID)
} // namespace ai_pass_selector
#endif // A3C_TRAINER_HPP
