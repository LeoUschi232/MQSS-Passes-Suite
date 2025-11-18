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
A3C_TCN::A3C_TCN(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  // Nr trainable parameters: unknown
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 5u;
  this->initialize(
      make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size),
      make_TCN_critic(max_qubits, nr_residual_blocks, kernel_size));
}

A3C_LSTM::A3C_LSTM(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  constexpr unsigned int hidden_size_multiplier = 5u;
  constexpr unsigned int projection_size_multiplier = 5u;
  this->initialize(make_LSTM_actor(max_qubits, hidden_size_multiplier,
                                   projection_size_multiplier),
                   make_LSTM_critic(max_qubits, hidden_size_multiplier,
                                    projection_size_multiplier));
}

A3C_HYBRID::A3C_HYBRID(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
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

std::unique_ptr<BaseA3CAgent> A3C_TCN::clone() const {
  return std::make_unique<A3C_TCN>(this->max_qubits, /*is_boss=*/false);
}

std::unique_ptr<BaseA3CAgent> A3C_LSTM::clone() const {
  return std::make_unique<A3C_LSTM>(this->max_qubits, /*is_boss=*/false);
}

std::unique_ptr<BaseA3CAgent> A3C_HYBRID::clone() const {
  return std::make_unique<A3C_HYBRID>(this->max_qubits, /*is_boss=*/false);
}

std::string A3C_TCN::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-tcn";
  return oss.str();
}

std::string A3C_LSTM::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-lstm";
  return oss.str();
}

std::string A3C_HYBRID::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-hybrid";
  return oss.str();
}
} // namespace ai_pass_selector
