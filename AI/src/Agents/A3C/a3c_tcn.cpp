#include "Agents/A3C/a3c_agents.hpp"

// Environment includes
#include "Environment/parallel_environments.hpp"
#include "Environment/quantum_circuit_environment.hpp"

// Utils includes
#include "Agents/agent_utils.hpp"
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <sstream>

namespace ai_pass_selector {

A3C_TCN::A3C_TCN(unsigned int max_qubits,
                 std::unordered_map<std::string, std::string> params,
                 bool is_boss)
    : BaseA3CAgent(max_qubits, std::move(params), is_boss) {
  // Treat the nr of neurons for an instruction representation as the nr of
  // input channels in a single unit of the chain.
  unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  unsigned int kernel_size = 5u;
  unsigned int padding = 2u;
  // Ignore this for now.
  auto actor = torch::nn::Sequential();
  auto critic = torch::nn::Sequential();
  this->initialize(actor, critic);
}

std::unique_ptr<BaseA3CAgent> A3C_TCN::clone() const {
  auto cloned = std::make_unique<A3C_TCN>(
      this->max_qubits, this->params_for_cloning, /*is_boss=*/false);
  return cloned;
}

std::string A3C_TCN::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-tcn";
  return oss.str();
}
} // namespace ai_pass_selector
