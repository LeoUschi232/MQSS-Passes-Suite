#include "NeuralNetworks/Agents/A3C/a3c_agents.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Agents includes
#include "NeuralNetworks/layers_and_networks.hpp"
#include "NeuralNetworks/tcn_full_network.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <sstream>

namespace ai_pass_selector {
A3C_TCN_RELU::A3C_TCN_RELU(unsigned int max_qubits, bool is_boss)
    : BaseA3CAgent(max_qubits, is_boss) {
  // Treat the nr of neurons for an instruction representation as the nr of
  // input channels in a single unit of the chain.
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  constexpr unsigned int nr_residual_blocks = 12u;
  constexpr unsigned int kernel_size = 5u;

  // Instructions Tensor will have shape [N, IRS]
  // N = Nr of instructions in the quantum circuit
  // IRS = Instruction Representation Size
  auto actor = torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                             kernel_size), // -> [IRS, N]
      torch::nn::AdaptiveAvgPool1d(1u),    // -> [IRS, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [NR_PASSES]
      torch::nn::Linear(IRS, NR_PASSES),                     // -> [NR_PASSES]
      torch::nn::ReLU(),                                     // -> [NR_PASSES]
      torch::nn::Softmax(/*dim=*/0u)                         // -> [NR_PASSES]
  );
  auto critic = torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                             kernel_size), // -> [IRS, N]
      torch::nn::AdaptiveAvgPool1d(1u),    // -> [IRS, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [NR_PASSES]
      torch::nn::Linear(IRS, 1),                             // -> [1]
      torch::nn::ReLU()                                      // -> [1]
  );
  this->BaseA3CAgent::initialize(actor, critic);
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
