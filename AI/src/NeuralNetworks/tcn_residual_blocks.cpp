#include "NeuralNetworks/Agents/tcn_residual_blocks.hpp"

// Agents includes
#include "NeuralNetworks/Agents/agent_layers_and_networks.hpp"

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
TCNResidualBlock::TCNResidualBlock(unsigned int in_channels,
                                   unsigned int out_channels,
                                   unsigned int kernel_size,
                                   unsigned int dilation)
    : weight_norm_conv1(torch::nn::WeightNormConv1d(in_channels, out_channels,
                                                    kernel_size, dilation)),
      weight_norm_conv2(torch::nn::WeightNormConv1d(out_channels, out_channels,
                                                    kernel_size, dilation)),
      downsample(in_channels != out_channels
                     ? torch::nn::Conv1d(torch::nn::Conv1dOptions(
                           in_channels, out_channels, 1))
                     : nullptr) {
  if (kernel_size % 2 == 0) {
    throw std::invalid_argument("Kernel size in TCNResidualBlock must be odd.");
  }
  this->register_module("weight_norm_conv1", this->weight_norm_conv1);
  this->register_module("weight_norm_conv2", this->weight_norm_conv2);
  if (this->downsample) {
    this->register_module("downsample", downsample);
    torch::NoGradGuard _;
    (void)this->downsample->weight.normal_(0, 0.01);
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
      this->weight_norm_conv1, // -> [C_out, N] // <- Here I want weight norm to
                               // remove layer norm
      torch::nn::ReLU(),       // -> [C_out, N]
      torch::nn::Dropout(
          torch::nn::DropoutOptions().p(dropout)), // -> [C_out, N]
      this->weight_norm_conv2,                     // -> [C_out, N]
      torch::nn::ReLU(),                           // -> [C_out, N]
      torch::nn::Dropout(
          torch::nn::DropoutOptions().p(dropout)) // -> [C_out, N]
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
                                                     double prelu_init,
                                                     double dropout)
    : TCNResidualBlock(in_channels, out_channels, kernel_size, dilation) {
  // C_in = number of input channels
  // C_out = number of output channels
  // N = sequence length
  // Expects input shape: [C_in, N]
  // Exerts output shape: [C_out, N]
  this->convolutional_block = torch::nn::Sequential(
      this->weight_norm_conv1, // -> [C_out, N]
      torch::nn::PReLU(
          torch::nn::PReLUOptions().init(prelu_init)), // -> [C_out, N]
      torch::nn::Dropout(
          torch::nn::DropoutOptions().p(dropout)), // -> [C_out, N]
      this->weight_norm_conv2,                     // -> [C_out, N]
      torch::nn::PReLU(
          torch::nn::PReLUOptions().init(prelu_init)), // -> [C_out, N]
      torch::nn::Dropout(
          torch::nn::DropoutOptions().p(dropout)) // -> [C_out, N]
  );
  this->final_prelu =
      torch::nn::PReLU(torch::nn::PReLUOptions().init(prelu_init));
  this->register_module("convolutional_block", convolutional_block);
  this->register_module("final_prelu", final_prelu);
}
torch::Tensor TCNResidualBlockWithPReLU::forward(const torch::Tensor &x) {
  return final_prelu->forward(
      this->convolutional_block->forward(x) +
      (this->downsample ? this->downsample->forward(x) : x));
}

} // namespace ai_pass_selector
