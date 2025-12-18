#include "NeuralNetworks/Agents/ACER/acer_agents.hpp"

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

ACER_TCN::ACER_TCN(unsigned int max_qubits) : BaseACERAgent(max_qubits) {
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 5u;
  this->initialize(
      make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size),
      make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size),
      make_TCN_Q_estimator(max_qubits, nr_residual_blocks, kernel_size));
}

ACER_LSTM::ACER_LSTM(unsigned int max_qubits) : BaseACERAgent(max_qubits) {
  constexpr unsigned int hidden_size_multiplier = 5u;
  constexpr unsigned int projection_size_multiplier = 5u;
  this->initialize(make_LSTM_actor(max_qubits, hidden_size_multiplier,
                                   projection_size_multiplier),
                   make_LSTM_actor(max_qubits, hidden_size_multiplier,
                                   projection_size_multiplier),
                   make_LSTM_Q_estimator(max_qubits, hidden_size_multiplier,
                                         projection_size_multiplier));
}

ACER_HYBRID::ACER_HYBRID(unsigned int max_qubits) : BaseACERAgent(max_qubits) {
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(HYBRID_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 3u;
  constexpr unsigned int lstm_hidden_size_multiplier = 3u;
  constexpr unsigned int lstm_projection_size_multiplier = 3u;
  this->initialize(make_hybrid_actor(max_qubits, nr_residual_blocks,
                                     kernel_size, lstm_hidden_size_multiplier,
                                     lstm_projection_size_multiplier),
                   make_hybrid_actor(max_qubits, nr_residual_blocks,
                                     kernel_size, lstm_hidden_size_multiplier,
                                     lstm_projection_size_multiplier),
                   make_hybrid_Q_estimator(max_qubits, nr_residual_blocks,
                                           kernel_size,
                                           lstm_hidden_size_multiplier,
                                           lstm_projection_size_multiplier));
}

std::string ACER_TCN::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "acer-" << size_string << "-tcn";
  return oss.str();
}

std::string ACER_LSTM::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "acer-" << size_string << "-lstm";
  return oss.str();
}

std::string ACER_HYBRID::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "acer-" << size_string << "-hybrid";
  return oss.str();
}

} // namespace ai_pass_selector
