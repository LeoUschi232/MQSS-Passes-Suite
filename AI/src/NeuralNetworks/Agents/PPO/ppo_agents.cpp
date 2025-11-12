#include "NeuralNetworks/Agents/PPO/ppo_agents.hpp"

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
PPO_TCN_RELU::PPO_TCN_RELU(unsigned int max_qubits) : BasePPOAgent(max_qubits) {
  // Nr trainable parameters: unknown
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 5u;
  this->BasePPOAgent::initialize(
      make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size),
      make_TCN_critic(max_qubits, nr_residual_blocks, kernel_size));
}

PPO_TCN_PRELU::PPO_TCN_PRELU(unsigned int max_qubits)
    : BasePPOAgent(max_qubits) {
  // Nr trainable parameters: unknown
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 5u;
  constexpr double prelu_init = 0.1;
  this->initialize(
      make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size, prelu_init),
      make_TCN_critic(max_qubits, nr_residual_blocks, kernel_size, prelu_init));
}

PPO_LSTM_HMPP::PPO_LSTM_HMPP(unsigned int max_qubits)
    : BasePPOAgent(max_qubits) {
  constexpr unsigned int hidden_size_multiplier = 8u;
  constexpr unsigned int projection_size_multiplier = 2u;
  this->initialize(make_LSTM_actor(max_qubits, hidden_size_multiplier,
                                   projection_size_multiplier),
                   make_LSTM_critic(max_qubits, hidden_size_multiplier,
                                    projection_size_multiplier));
}

PPO_LSTM_BMNP::PPO_LSTM_BMNP(unsigned int max_qubits)
    : BasePPOAgent(max_qubits) {
  constexpr unsigned int hidden_size_multiplier = 5u;
  constexpr unsigned int projection_size_multiplier = 5u;
  this->initialize(make_LSTM_actor(max_qubits, hidden_size_multiplier,
                                   projection_size_multiplier),
                   make_LSTM_critic(max_qubits, hidden_size_multiplier,
                                    projection_size_multiplier));
}

PPO_HYBRID::PPO_HYBRID(unsigned int max_qubits) : BasePPOAgent(max_qubits) {
  // Nr trainable parameters: unknown
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(HYBRID_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 3u;
  constexpr unsigned int hidden_size_multiplier = 3u;
  constexpr unsigned int projection_size_multiplier = 3u;
  this->initialize(
      make_hybrid_actor(max_qubits, nr_residual_blocks, kernel_size,
                        hidden_size_multiplier, projection_size_multiplier),
      make_hybrid_critic(max_qubits, nr_residual_blocks, kernel_size,
                         hidden_size_multiplier, projection_size_multiplier));
}

std::string PPO_TCN_RELU::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "ppo-" << size_string << "-tcnrelu";
  return oss.str();
}

std::string PPO_TCN_PRELU::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "ppo-" << size_string << "-tcnprelu";
  return oss.str();
}

std::string PPO_LSTM_HMPP::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "ppo-" << size_string << "-lstmhmpp";
  return oss.str();
}

std::string PPO_LSTM_BMNP::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "ppo-" << size_string << "-lstmbmnp";
  return oss.str();
}

std::string PPO_HYBRID::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "ppo-" << size_string << "-hybrid";
  return oss.str();
}
} // namespace ai_pass_selector
