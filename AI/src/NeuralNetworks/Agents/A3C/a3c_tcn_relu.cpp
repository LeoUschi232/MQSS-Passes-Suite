#include "NeuralNetworks/Agents/A3C/a3c_agents.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Agents includes
#include "NeuralNetworks/layers_and_networks.hpp"

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

std::unique_ptr<BaseA3CAgent> A3C_TCN_RELU::clone() const {
  return std::make_unique<A3C_TCN_RELU>(this->max_qubits, /*is_boss=*/false);
}

std::string A3C_TCN_RELU::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-tcnrelu";
  return oss.str();
}
} // namespace ai_pass_selector
