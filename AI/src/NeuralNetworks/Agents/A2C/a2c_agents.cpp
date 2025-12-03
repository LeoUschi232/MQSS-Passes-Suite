#include "NeuralNetworks/Agents/A2C/a2c_agents.hpp"

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
A2C_TCN::A2C_TCN(unsigned int max_qubits) : BaseA2CAgent(max_qubits) {
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 5u;
  this->initialize(
      make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size),
      make_TCN_critic(max_qubits, nr_residual_blocks, kernel_size));
}

A2C_LSTM::A2C_LSTM(unsigned int max_qubits) : BaseA2CAgent(max_qubits) {
  constexpr unsigned int hidden_size_multiplier = 5u;
  constexpr unsigned int projection_size_multiplier = 5u;
  this->initialize(make_LSTM_actor(max_qubits, hidden_size_multiplier,
                                   projection_size_multiplier),
                   make_LSTM_critic(max_qubits, hidden_size_multiplier,
                                    projection_size_multiplier));
}

A2C_HYBRID::A2C_HYBRID(unsigned int max_qubits) : BaseA2CAgent(max_qubits) {
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

std::string A2C_TCN::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a2c-" << size_string << "-tcn";
  return oss.str();
}

std::string A2C_LSTM::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a2c-" << size_string << "-lstm";
  return oss.str();
}

std::string A2C_HYBRID::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a2c-" << size_string << "-hybrid";
  return oss.str();
}
} // namespace ai_pass_selector
