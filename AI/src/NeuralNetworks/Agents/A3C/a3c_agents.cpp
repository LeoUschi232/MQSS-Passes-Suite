#include "NeuralNetworks/Agents/A3C/a3c_agents.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Agents includes
#include "NeuralNetworks/agent_architectures.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <cmath>
#include <sstream>

namespace ai_pass_selector {
A3C_TCN_RELU::A3C_TCN_RELU(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  // Nr trainable parameters: unknown
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 5u;
  this->BaseA3CAgent::initialize(
      make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size),
      make_TCN_critic(max_qubits, nr_residual_blocks, kernel_size));
}

A3C_TCN_PRELU::A3C_TCN_PRELU(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  // Nr trainable parameters: unknown
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 5u;
  constexpr double prelu_init = 0.1;
  this->BaseA3CAgent::initialize(
      make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size, prelu_init),
      make_TCN_critic(max_qubits, nr_residual_blocks, kernel_size, prelu_init));
}

A3C_LSTM_HMPP::A3C_LSTM_HMPP(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  // Nr trainable parameters: 5,111,487
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
  // Nr trainable parameters: 5,542,587
  constexpr unsigned int hidden_size_multiplier = 5u;
  constexpr unsigned int projection_size_multiplier = 5u;
  this->BaseA3CAgent::initialize(
      make_LSTM_actor(max_qubits, hidden_size_multiplier,
                      projection_size_multiplier),
      make_LSTM_critic(max_qubits, hidden_size_multiplier,
                       projection_size_multiplier));
}

A3C_HYBRID::A3C_HYBRID(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  // Nr trainable parameters: unknown
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(HYBRID_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 3u;
  constexpr unsigned int hidden_size_multiplier = 3u;
  constexpr unsigned int projection_size_multiplier = 3u;
  this->BaseA3CAgent::initialize(
      make_hybrid_actor(max_qubits, nr_residual_blocks, kernel_size,
                        hidden_size_multiplier, projection_size_multiplier),
      make_hybrid_critic(max_qubits, nr_residual_blocks, kernel_size,
                         hidden_size_multiplier, projection_size_multiplier));
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

std::unique_ptr<BaseA3CAgent> A3C_HYBRID::clone() const {
  return std::make_unique<A3C_HYBRID>(this->max_qubits, /*is_boss=*/false);
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

std::string A3C_HYBRID::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-hybrid";
  return oss.str();
}
} // namespace ai_pass_selector
