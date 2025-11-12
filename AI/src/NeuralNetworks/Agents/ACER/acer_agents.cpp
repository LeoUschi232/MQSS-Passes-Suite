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

ACER_TCN_RELU::ACER_TCN_RELU(unsigned int max_qubits)
    : BaseACERAgent(max_qubits) {
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 5u;
  this->initialize(
      /*actor=*/make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size),
      /*critic_Q_estimator=*/
      make_TCN_Q_estimator(max_qubits, nr_residual_blocks, kernel_size));
}
ACER_TCN_PRELU::ACER_TCN_PRELU(unsigned int max_qubits)
    : BaseACERAgent(max_qubits) {
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 5u;
  constexpr double prelu_init = 0.1;
  this->initialize(
      /*actor=*/make_TCN_actor(max_qubits, nr_residual_blocks, kernel_size,
                               prelu_init),
      /*critic_Q_estimator=*/
      make_TCN_Q_estimator(max_qubits, nr_residual_blocks, kernel_size,
                           prelu_init));
}

ACER_LSTM_HMPP::ACER_LSTM_HMPP(unsigned int max_qubits)
    : BaseACERAgent(max_qubits) {
  constexpr unsigned int hidden_size_multiplier = 8u;
  constexpr unsigned int projection_size_multiplier = 2u;
  this->initialize(
      /*actor=*/make_LSTM_actor(max_qubits, hidden_size_multiplier,
                                projection_size_multiplier),
      /*critic_Q_estimator=*/
      make_LSTM_Q_estimator(max_qubits, hidden_size_multiplier,
                            projection_size_multiplier));
}

ACER_LSTM_BMNP::ACER_LSTM_BMNP(unsigned int max_qubits)
    : BaseACERAgent(max_qubits) {
  constexpr unsigned int hidden_size_multiplier = 5u;
  constexpr unsigned int projection_size_multiplier = 5u;
  this->initialize(
      /*actor=*/make_LSTM_actor(max_qubits, hidden_size_multiplier,
                                projection_size_multiplier),
      /*critic_Q_estimator=*/
      make_LSTM_Q_estimator(max_qubits, hidden_size_multiplier,
                            projection_size_multiplier));
}

ACER_HYBRID::ACER_HYBRID(unsigned int max_qubits) : BaseACERAgent(max_qubits) {
  const unsigned int nr_residual_blocks =
      std::ceil(std::log2(TCN_QUBIT_MAGIC * max_qubits));
  constexpr unsigned int kernel_size = 3u;
  constexpr unsigned int lstm_hidden_size_multiplier = 3u;
  constexpr unsigned int lstm_projection_size_multiplier = 3u;
  this->initialize(
      /*actor_main=*/make_hybrid_actor(max_qubits, nr_residual_blocks, kernel_size,
                                  lstm_hidden_size_multiplier,
                                  lstm_projection_size_multiplier),
      /*actor=*/make_hybrid_actor(max_qubits, nr_residual_blocks, kernel_size,
                                  lstm_hidden_size_multiplier,
                                  lstm_projection_size_multiplier),
      /*critic_Q_estimator=*/
      make_hybrid_Q_estimator(max_qubits, nr_residual_blocks, kernel_size,
                              lstm_hidden_size_multiplier,
                              lstm_projection_size_multiplier));
}

std::string ACER_TCN_RELU::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "acer-" << size_string << "-tcnrelu";
  return oss.str();
}

std::string ACER_TCN_PRELU::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "acer-" << size_string << "-tcnprelu";
  return oss.str();
}

std::string ACER_LSTM_HMPP::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "acer-" << size_string << "-lstmhmpp";
  return oss.str();
}

std::string ACER_LSTM_BMNP::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "acer-" << size_string << "-lstmbmnp";
  return oss.str();
}

std::string ACER_HYBRID::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "acer-" << size_string << "-hybrid";
  return oss.str();
}

} // namespace ai_pass_selector
