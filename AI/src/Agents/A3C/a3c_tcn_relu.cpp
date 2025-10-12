#include "Agents/A3C/a3c_agents.hpp"

// Environment includes
#include "Environment/parallel_environments.hpp"
#include "Environment/quantum_circuit_environment.hpp"

// Agents includes
#include "Agents/agent_layers_and_networks.hpp"
#include "Agents/agent_utils.hpp"
#include "Agents/tcn_full_network.hpp"

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
  constexpr unsigned int inner_kernel_size = 5u;
  constexpr unsigned int final_kernel_size = 1u;

  // Instructions Tensor will have shape [N, IRS]
  // N = Nr of instructions in the quantum circuit
  // IRS = Instruction Representation Size
  auto actor = torch::nn::Sequential(
      torch::nn::ShapeProbe("actor_on_input"),
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      torch::nn::ShapeProbe("actor_after_transpose"),
      TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                             inner_kernel_size), // -> [IRS, N]
      torch::nn::ShapeProbe("actor_after_tcn"),
      torch::nn::Conv1d(torch::nn::Conv1dOptions(
          IRS, NR_PASSES, final_kernel_size)), // -> [NR_PASSES, N]
      torch::nn::ShapeProbe("actor_after_final_conv"),
      torch::nn::ReLU(), // -> [NR_PASSES, N]
      torch::nn::ShapeProbe("actor_after_final_relu"),
      torch::nn::AdaptiveAvgPool1d(1u), // -> [NR_PASSES, 1]
      torch::nn::ShapeProbe("actor_after_avg_pool"),
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [NR_PASSES]
      torch::nn::Softmax(/*dim=*/0u)                         // -> [NR_PASSES]
  );
  auto critic = torch::nn::Sequential(
      torch::nn::ShapeProbe("critic_on_input"),
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      torch::nn::ShapeProbe("critic_after_transpose"),
      TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                             inner_kernel_size), // -> [IRS, N]
      torch::nn::ShapeProbe("critic_after_tcn"),
      torch::nn::Conv1d(
          torch::nn::Conv1dOptions(IRS, 1u, final_kernel_size)), // -> [1, N]
      torch::nn::ShapeProbe("critic_after_final_conv"),
      torch::nn::ReLU(), // -> [1, N]
      torch::nn::ShapeProbe("critic_after_final_relu"),
      torch::nn::AdaptiveAvgPool1d(1u), // -> [1, 1]
      torch::nn::ShapeProbe("critic_after_avg_pool"),
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)) // -> [1]
  );
  this->initialize(actor, critic);
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
