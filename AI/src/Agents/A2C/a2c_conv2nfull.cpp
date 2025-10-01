#include "Agents/A2C/a2c_agents.hpp"

// Environment includes
#include "Environment/environment.hpp"

// Torch includes
#include "../../../include/Environment/parallel_environments.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <sstream>

namespace ai_pass_selector {

A2C_CONV2NFULL::A2C_CONV2NFULL(
    unsigned int max_qubits,
    std::unordered_map<std::string, std::string> params)
    : BaseA2CAgent(max_qubits, std::move(params)) {
  // Treat the nr of neurons for an instruction representation as the nr of
  // input channels in a single unit of the chain.
  unsigned int IRP = MAX_QUBITS_TO_INSTRUCTION_REPRESENTATION_SIZE(max_qubits);

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
      torch::nn::Conv1d(torch::nn::Conv1dOptions(IRP, IRP, max_qubits)
                            .padding(padding)), // Shape {B, IRP, N}
      torch::nn::GroupNorm(
          torch::nn::GroupNormOptions(1, IRP)), // Shape {B, IRP, N}
      torch::nn::HalfScalingLayer(),            // Shape {B, IRP, N}

      // Inner layer nr 2
      torch::nn::Conv1d(torch::nn::Conv1dOptions(IRP, IRP, max_qubits)
                            .padding(padding)), // Shape {B, IRP, N}
      torch::nn::GroupNorm(
          torch::nn::GroupNormOptions(1, IRP)), // Shape {B, IRP, N}
      torch::nn::HalfScalingLayer(),            // Shape {B, IRP, N}

      // Output layer
      torch::nn::Conv1d(torch::nn::Conv1dOptions(IRP, NR_PASSES, max_qubits)
                            .padding(padding)), // Shape {B, NR_PASSES, N}
      torch::nn::AdaptiveAvgPool1d(1),          // Shape {B, NR_PASSES, 1}
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(1)), // Shape {B, NR_PASSES}
      torch::nn::Softmax(torch::nn::SoftmaxOptions(/*dim=*/1)));
  auto critic = torch::nn::Sequential(
      // Input layer
      torch::nn::TransposeContiguous(1, 2), // Shape {B, IRP, N}

      // Inner layer nr 1
      torch::nn::Conv1d(torch::nn::Conv1dOptions(IRP, IRP, max_qubits)
                            .padding(padding)), // Shape {B, IRP, N}
      torch::nn::GroupNorm(
          torch::nn::GroupNormOptions(1, IRP)), // Shape {B, IRP, N}
      torch::nn::HalfScalingLayer(),            // Shape {B, IRP, N}

      // Inner layer nr 2
      torch::nn::Conv1d(torch::nn::Conv1dOptions(IRP, IRP, max_qubits)
                            .padding(padding)), // Shape {B, IRP, N}
      torch::nn::GroupNorm(
          torch::nn::GroupNormOptions(1, IRP)), // Shape {B, IRP, N}
      torch::nn::HalfScalingLayer(),            // Shape {B, IRP, N}

      // Output layer
      torch::nn::Conv1d(torch::nn::Conv1dOptions(IRP, 1, max_qubits)
                            .padding(padding)), // Shape {B, 1, N}
      torch::nn::AdaptiveAvgPool1d(1),          // Shape {B, 1, 1}
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(1)) // Shape {B, 1}
  );
  this->initialize(actor, critic);
}

std::string A2C_CONV2NFULL::agentName() const {
  std::string size_string = "mq" + std::to_string(this->max_qubits);
  std::ostringstream oss;
  oss << "a2c-" << size_string << "-conv2nfull";
  return oss.str();
}
} // namespace ai_pass_selector
