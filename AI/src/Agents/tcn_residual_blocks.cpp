#include "Agents/tcn_residual_blocks.hpp"

// Agents includes
#include "Agents/agent_layers_and_networks.hpp"

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
TCNResidualBlock::TCNResidualBlock(unsigned int in_channels,
                                   unsigned int out_channels,
                                   unsigned int kernel_size,
                                   unsigned int dilation)
    : conv1(torch::nn::Conv1d(
          torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
              .dilation(dilation)
              .padding(dilation * (kernel_size - 1) / 2))),
      conv2(torch::nn::Conv1d(
          torch::nn::Conv1dOptions(out_channels, out_channels, kernel_size)
              .dilation(dilation)
              .padding(dilation * (kernel_size - 1) / 2))),
      downsample(in_channels != out_channels
                     ? torch::nn::Conv1d(torch::nn::Conv1dOptions(
                           in_channels, out_channels, 1))
                     : nullptr) {
  if (kernel_size % 2 == 0) {
    throw std::invalid_argument("Kernel size in TCNResidualBlock must be odd.");
  }

  this->init_weights();
  this->register_module("conv1", conv1);
  this->register_module("conv2", conv2);
  if (this->downsample) {
    this->register_module("downsample", downsample);
  }
}
void TCNResidualBlock::init_weights() {
  (void)this->conv1->weight.data().normal_(0, 0.01);
  (void)this->conv2->weight.data().normal_(0, 0.01);
  if (this->downsample) {
    (void)this->downsample->weight.data().normal_(0, 0.01);
  }
}
TCNResidualBlockWithReLU::TCNResidualBlockWithReLU(unsigned int in_channels,
                                                   unsigned int out_channels,
                                                   unsigned int kernel_size,
                                                   unsigned int dilation,
                                                   double dropout)
    : TCNResidualBlock(in_channels, out_channels, kernel_size, dilation) {
  // C_in = number of input channels
  // C_out = number of output channels
  // N = sequence length
  // Expects input shape: [C_in, N]
  // Exerts output shape: [C_out, N]
  this->convolutional_block = torch::nn::Sequential(
      this->conv1,                // -> [C_out, N] // <- Here I want weight norm to remove layer norm
      torch::nn::Transpose(0, 1), // -> [N, C_out]
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({out_channels})), // -> [N, C_out]
      torch::nn::ReLU(),                                // -> [N, C_out]
      torch::nn::Dropout(
          torch::nn::DropoutOptions().p(dropout)), // -> [N, C_out]
      torch::nn::TransposeContiguous(0, 1),        // -> [C_out, N]
      this->conv2,                                 // -> [C_out, N]
      torch::nn::Transpose(0, 1),                  // -> [N, C_out]
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({out_channels})), // -> [N, C_out]
      torch::nn::ReLU(),                                // -> [N, C_out]
      torch::nn::Dropout(
          torch::nn::DropoutOptions().p(dropout)), // -> [N, C_out]
      torch::nn::TransposeContiguous(0, 1)         // -> [C_out, N]
  );
  this->final_relu = torch::nn::ReLU();
  this->register_module("convolutional_block", convolutional_block);
  this->register_module("final_relu", final_relu);
}
torch::Tensor TCNResidualBlockWithReLU::forward(const torch::Tensor &x) {
  return final_relu->forward(
      this->convolutional_block->forward(x) +
      (this->downsample ? this->downsample->forward(x) : x));
}

TCNResidualBlockWithPReLU::TCNResidualBlockWithPReLU(unsigned int in_channels,
                                                     unsigned int out_channels,
                                                     unsigned int kernel_size,
                                                     unsigned int dilation,
                                                     double dropout,
                                                     double final_prelu_init)
    : TCNResidualBlock(in_channels, out_channels, kernel_size, dilation) {
  // C_in = number of input channels
  // C_out = number of output channels
  // N = sequence length
  // Expects input shape: [C_in, N]
  // Exerts output shape: [C_out, N]
  this->convolutional_block = torch::nn::Sequential(
      this->conv1,                // -> [C_out, N]
      torch::nn::Transpose(0, 1), // -> [N, C_out]
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({out_channels})), // -> [N, C_out]
      torch::nn::PReLU(
          torch::nn::PReLUOptions().init(final_prelu_init)), // -> [N, C_out]
      torch::nn::Dropout(
          torch::nn::DropoutOptions().p(dropout)), // -> [N, C_out]
      torch::nn::TransposeContiguous(0, 1),        // -> [C_out, N]
      this->conv2,                                 // -> [C_out, N]
      torch::nn::Transpose(0, 1),                  // -> [N, C_out]
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({out_channels})), // -> [N, C_out]
      torch::nn::PReLU(
          torch::nn::PReLUOptions().init(final_prelu_init)), // -> [N, C_out]
      torch::nn::Dropout(
          torch::nn::DropoutOptions().p(dropout)), // -> [N, C_out]
      torch::nn::TransposeContiguous(0, 1)         // -> [C_out, N]
  );
  this->final_prelu =
      torch::nn::PReLU(torch::nn::PReLUOptions().init(final_prelu_init));
  this->register_module("convolutional_block", convolutional_block);
  this->register_module("final_prelu", final_prelu);
}
torch::Tensor TCNResidualBlockWithPReLU::forward(const torch::Tensor &x) {
  return final_prelu->forward(
      this->convolutional_block->forward(x) +
      (this->downsample ? this->downsample->forward(x) : x));
}

} // namespace ai_pass_selector
