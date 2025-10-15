#include "NeuralNetworks/Agents/A3C/a3c_agents.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Agents includes
#include "NeuralNetworks/agent_architectures.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <sstream>

namespace ai_pass_selector {
A3C_TCN_RELU::A3C_TCN_RELU(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  this->BaseA3CAgent::initialize(make_TCN_actor(max_qubits),
                                 make_TCN_critic(max_qubits));
}

A3C_TCN_PRELU::A3C_TCN_PRELU(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  constexpr double prelu_init = 0.1;
  this->BaseA3CAgent::initialize(make_TCN_actor(max_qubits, prelu_init),
                                 make_TCN_critic(max_qubits, prelu_init));
}

A3C_LSTM_HMPP::A3C_LSTM_HMPP(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  // MQTBench case: IRS=150
  // H=8·IRS=1200
  // P=2·IRS=300
  // D=2
  // → Nr trainable parameters: 5,111,487
  constexpr unsigned int hidden_size_multiplier = 8u;
  constexpr unsigned int projection_size_multiplier = 2u;
  this->BaseA3CAgent::initialize(
      make_LSTM_actor(max_qubits, hidden_size_multiplier,
                      projection_size_multiplier),
      make_LSTM_critic(max_qubits, hidden_size_multiplier,
                       projection_size_multiplier));
}

A3C_LSTM_BMNP::A3C_LSTM_BMNP(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  // MQTBench case: IRS=150
  // H=5·IRS=750
  // P=5·IRS=750
  // D=2
  // → Nr trainable parameters: 5,542,587
  constexpr unsigned int hidden_size_multiplier = 5u;
  constexpr unsigned int projection_size_multiplier = 5u;
  this->BaseA3CAgent::initialize(
      make_LSTM_actor(max_qubits, hidden_size_multiplier,
                      projection_size_multiplier),
      make_LSTM_critic(max_qubits, hidden_size_multiplier,
                       projection_size_multiplier));
}

std::unique_ptr<BaseA3CAgent> A3C_TCN_RELU::clone() const {
  return std::make_unique<A3C_TCN_RELU>(this->max_qubits, /*is_boss=*/false);
}

std::unique_ptr<BaseA3CAgent> A3C_TCN_PRELU::clone() const {
  return std::make_unique<A3C_TCN_PRELU>(this->max_qubits, /*is_boss=*/false);
}

std::unique_ptr<BaseA3CAgent> A3C_LSTM_HMPP::clone() const {
  return std::make_unique<A3C_LSTM_HMPP>(this->max_qubits, /*is_boss=*/false);
}

std::unique_ptr<BaseA3CAgent> A3C_LSTM_BMNP::clone() const {
  return std::make_unique<A3C_LSTM_BMNP>(this->max_qubits, /*is_boss=*/false);
}

std::string A3C_TCN_RELU::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-tcnrelu";
  return oss.str();
}

std::string A3C_TCN_PRELU::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-tcnprelu";
  return oss.str();
}

std::string A3C_LSTM_HMPP::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-lstmhmpp";
  return oss.str();
}

std::string A3C_LSTM_BMNP::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-lstmbmnp";
  return oss.str();
}
} // namespace ai_pass_selector
