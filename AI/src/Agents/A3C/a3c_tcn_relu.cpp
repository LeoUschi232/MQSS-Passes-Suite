#include "Agents/A3C/a3c_agents.hpp"

// Environment includes
#include "Environment/parallel_environments.hpp"
#include "Environment/quantum_circuit_environment.hpp"

// Utils includes
#include "Agents/agent_layers_and_networks.hpp"
#include "Agents/agent_utils.hpp"
#include "Agents/tcn_full_network.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <sstream>

namespace ai_pass_selector {
A3C_TCN_RELU::A3C_TCN_RELU(unsigned int max_qubits,
                           std::unordered_map<std::string, std::string> params,
                           bool is_boss)
    : BaseA3CAgent(max_qubits, std::move(params), is_boss) {
  // Treat the nr of neurons for an instruction representation as the nr of
  // input channels in a single unit of the chain.
  unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  unsigned int nr_residual_blocks = 12u;
  unsigned int inner_kernel_size = 5u;
  unsigned int final_kernel_size = 1u;

  // Instructions Tensor will have shape [N, IRS]
  // N = Nr of instructions in the quantum circuit
  // IRS = Instruction Representation Size
  auto actor = torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                             inner_kernel_size), // -> [IRS, N]
      torch::nn::Conv1d(torch::nn::Conv1dOptions(
          IRS, NR_PASSES, final_kernel_size)), // -> [NR_PASSES, N]
      torch::nn::ReLU(),                       // -> [NR_PASSES, N]
      torch::nn::AdaptiveAvgPool1d(1u),        // -> [NR_PASSES, 1]
      torch::nn::Flatten(),                    // -> [NR_PASSES]
      torch::nn::Softmax(/*dim=*/0u)           // -> [NR_PASSES]
  );
  auto critic = torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                             inner_kernel_size), // -> [IRS, N]
      torch::nn::Conv1d(
          torch::nn::Conv1dOptions(IRS, 1u, final_kernel_size)), // -> [1, N]
      torch::nn::ReLU(),                                         // -> [1, N]
      torch::nn::AdaptiveAvgPool1d(1u),                          // -> [1, 1]
      torch::nn::Flatten()                                       // -> [1]
  );
  this->initialize(actor, critic);
}

std::unique_ptr<BaseA3CAgent> A3C_TCN_RELU::clone() const {
  return std::make_unique<A3C_TCN_RELU>(
      this->max_qubits, this->params_for_cloning, /*is_boss=*/false);
}

std::string A3C_TCN_RELU::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a3c-" << size_string << "-tcnrelu";
  return oss.str();
}
} // namespace ai_pass_selector
