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

A3C_TCN_PRELU::A3C_TCN_PRELU(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  // Input shape: [N, IRS]
  // N = Nr of instructions in the quantum circuit
  // IRS = Instruction Representation Size
  constexpr double prelu_init = 0.1;
  this->BaseA3CAgent::initialize(make_TCN_actor(max_qubits, prelu_init),
                                 make_TCN_critic(max_qubits, prelu_init));
}

std::unique_ptr<BaseA3CAgent> A3C_TCN_PRELU::clone() const {
  return std::make_unique<A3C_TCN_PRELU>(this->max_qubits, /*is_boss=*/false);
}

std::string A3C_TCN_PRELU::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-tcnprelu";
  return oss.str();
}
} // namespace ai_pass_selector
