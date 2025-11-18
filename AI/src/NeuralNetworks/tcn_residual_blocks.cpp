#include "NeuralNetworks/tcn_residual_blocks.hpp"

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
TCNResidualBlock::TCNResidualBlock(unsigned int in_channels,
                                   unsigned int out_channels,
                                   unsigned int kernel_size,
                                   unsigned int dilation,
                                   double dropout)
    : convolutional_block( // Input: [C_in, N]
          torch::nn::WeightNormConv1d(in_channels, out_channels, kernel_size,
                                      dilation), // -> [C_out, N]
          torch::nn::ReLU(),                     // -> [C_out, N]
          torch::nn::Dropout(
              torch::nn::DropoutOptions().p(dropout)), // -> [C_out, N]
          torch::nn::WeightNormConv1d(out_channels, out_channels, kernel_size,
                                      dilation), // -> [C_out, N]
          torch::nn::ReLU(),                     // -> [C_out, N]
          torch::nn::Dropout(
              torch::nn::DropoutOptions().p(dropout)) // -> [C_out, N]
          ),
      layer_norm(torch::nn::Sequential(         // Input shape: [C_out, N]
          torch::nn::TransposeContiguous(0, 1), // -> [N, C_out]
          torch::nn::LayerNorm(torch::nn::LayerNormOptions(
              /*normalized_shape=*/{out_channels})), // -> [N, C_out]
          torch::nn::TransposeContiguous(0, 1)       // -> [C_out, N]
          )) {
  if (kernel_size % 2 == 0) {
    throw std::invalid_argument("Kernel size in TCNResidualBlock must be odd.");
  }
  this->register_module("convolutional_block", convolutional_block);
  this->register_module("layer_norm_block", layer_norm);
}

torch::Tensor TCNResidualBlock::forward(const torch::Tensor &x) {
  return this->layer_norm->forward(this->convolutional_block->forward(x) + x);
}
} // namespace ai_pass_selector
