#include "Agents/A2C/a2c_agents.hpp"

// Environment includes
#include "Environment/environment.hpp"
#include "Environment/parallel_environments.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <sstream>

namespace ai_pass_selector {

A2C_CONV3::A2C_CONV3(unsigned int max_qubits,
                           std::unordered_map<std::string, std::string> params)
    : BaseA2CAgent(max_qubits, std::move(params)) {
  // Treat the nr of neurons for an instruction representation as the nr of
  // input channels in a single unit of the chain.
  unsigned int IRP = MAX_QUBITS_TO_INSTRUCTION_REPRESENTATION_SIZE(max_qubits);

  // Reduce the dimensionality of the inner layers.
  // Standard practice in convolutional networks.
  // But keep the nr of channels always at least more than 1 and strictly
  // decreasing.
  unsigned int L2 = static_cast<unsigned>(3.0 / 4.0 * IRP);
  L2 = std::max(4u, L2);
  unsigned int L3 = static_cast<unsigned>(2.0 / 4.0 * IRP);
  L3 = std::max(3u, L3);
  unsigned int L4 = static_cast<unsigned>(1.0 / 4.0 * IRP);
  L4 = std::max(2u, L4);

  // Make sure there are at least half the kernel size in padding on each size
  // of the instruction chain so that even when there are 0 instructions, at
  // least 1 kernel slide is possible.
  unsigned int padding = max_qubits / 2;

  // Input tensor shape is {B, N, IRP} but a convolutional layer expects the
  // number of channels in each position before the nr of positions, so before
  // passing the input tensor to the neural network, it must be transformed to
  // shape {B, IRP, N}.
  auto actor = torch::nn::Sequential(
      // Input layer
      torch::nn::TransposeContiguous(1, 2), // Shape {B, IRP, N}

      // Inner layer nr 1
      torch::nn::Conv1d(torch::nn::Conv1dOptions(IRP, L2, max_qubits)
                            .padding(padding)), // Shape {B, L2, N}
      torch::nn::Transpose(1, 2),               // Shape {B, N, L2}
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({L2})), // Shape {B, N, L2}
      torch::nn::Transpose(1, 2),             // Shape {B, L2, N}
      torch::nn::HalfScalingLayer(),          // Shape {B, L2, N}

      // Inner layer nr 2
      torch::nn::Conv1d(torch::nn::Conv1dOptions(L2, L3, max_qubits)
                            .padding(padding)), // Shape {B, L3, N}
      torch::nn::Transpose(1, 2),               // Shape {B, N, L3}
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({L3})), // Shape {B, N, L3}
      torch::nn::Transpose(1, 2),             // Shape {B, L3, N}
      torch::nn::HalfScalingLayer(),          // Shape {B, L3, N}

      // Inner layer nr 3
      torch::nn::Conv1d(torch::nn::Conv1dOptions(L3, L4, max_qubits)
                            .padding(padding)), // Shape {B, L4, N}
      torch::nn::Transpose(1, 2),               // Shape {B, N, L4}
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({L4})), // Shape {B, N, L4}
      torch::nn::Transpose(1, 2),             // Shape {B, L4, N}
      torch::nn::HalfScalingLayer(),          // Shape {B, L4, N}

      // Output layer
      torch::nn::Conv1d(torch::nn::Conv1dOptions(L4, NR_PASSES, max_qubits)
                            .padding(padding)), // Shape {B, NR_PASSES, N}
      torch::nn::AdaptiveAvgPool1d(1),          // Shape {B, NR_PASSES, 1}
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(1)), // Shape {B, NR_PASSES}
      torch::nn::Softmax(torch::nn::SoftmaxOptions(/*dim=*/1)));
  auto critic = torch::nn::Sequential(
      // Input layer
      torch::nn::TransposeContiguous(1, 2), // Shape {B, IRP, N}

      // Inner layer nr 1
      torch::nn::Conv1d(torch::nn::Conv1dOptions(IRP, L2, max_qubits)
                            .padding(padding)), // Shape {B, L2, N}
      torch::nn::Transpose(1, 2),               // Shape {B, N, L2}
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({L2})), // Shape {B, N, L2}
      torch::nn::Transpose(1, 2),             // Shape {B, L2, N}
      torch::nn::HalfScalingLayer(),          // Shape {B, L2, N}

      // Inner layer nr 2
      torch::nn::Conv1d(torch::nn::Conv1dOptions(L2, L3, max_qubits)
                            .padding(padding)), // Shape {B, L3, N}
      torch::nn::Transpose(1, 2),               // Shape {B, N, L3}
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({L3})), // Shape {B, N, L3}
      torch::nn::Transpose(1, 2),             // Shape {B, L3, N}
      torch::nn::HalfScalingLayer(),          // Shape {B, L3, N}

      // Inner layer nr 3
      torch::nn::Conv1d(torch::nn::Conv1dOptions(L3, L4, max_qubits)
                            .padding(padding)), // Shape {B, L4, N}
      torch::nn::Transpose(1, 2),               // Shape {B, N, L4}
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({L4})), // Shape {B, N, L4}
      torch::nn::Transpose(1, 2),             // Shape {B, L4, N}
      torch::nn::HalfScalingLayer(),          // Shape {B, L4, N}

      // Output layer
      torch::nn::Conv1d(torch::nn::Conv1dOptions(L4, 1, max_qubits)
                            .padding(padding)), // Shape {B, 1, N}
      torch::nn::AdaptiveAvgPool1d(1),          // Shape {B, 1, 1}
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(1)) // Shape {B, 1}
  );
  this->initialize(actor, critic);
}

std::string A2C_CONV3::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a2c-" << size_string << "-conv3";
  return oss.str();
}
} // namespace ai_pass_selector
