#include "NeuralNetworks/tcn_network.hpp"

// Neural-Network includes
#include "NeuralNetworks/layers_and_wrappers.hpp"

// Torch includes
#include "torch/torch.h"

namespace torch::nn {
TCNResidualBlock::TCNResidualBlock(unsigned int in_channels,
                                   unsigned int out_channels,
                                   unsigned int kernel_size,
                                   unsigned int dilation,
                                   double dropout)
    : convolutional_block( // Input: [C_in, N]
          WeightNormConv1d(in_channels, out_channels, kernel_size,
                           dilation),           // -> [C_out, N]
          ReLU(),                               // -> [C_out, N]
          Dropout(DropoutOptions().p(dropout)), // -> [C_out, N]
          WeightNormConv1d(out_channels, out_channels, kernel_size,
                           dilation),          // -> [C_out, N]
          ReLU(),                              // -> [C_out, N]
          Dropout(DropoutOptions().p(dropout)) // -> [C_out, N]
          ),
      layer_norm(Sequential(         // Input shape: [C_out, N]
          TransposeContiguous(0, 1), // -> [N, C_out]
          LayerNorm(LayerNormOptions({out_channels})), // -> [N, C_out]
          TransposeContiguous(0, 1)                    // -> [C_out, N]
          )) {
  if (kernel_size % 2 == 0) {
    throw std::invalid_argument("Kernel size in TCNResidualBlock must be odd.");
  }
  this->register_module("convolutional_block", this->convolutional_block);
  this->register_module("layer_norm_block", this->layer_norm);
}

Tensor TCNResidualBlock::forward(const Tensor &x) {
  return this->layer_norm->forward(this->convolutional_block->forward(x) + x);
}
TCNFullNetwork::TCNFullNetwork(unsigned int nr_channels,
                               unsigned int nr_residual_blocks,
                               unsigned int kernel_size, double dropout)
    : tcn_sequence(Sequential()) {
  if (kernel_size % 2 == 0) {
    throw std::invalid_argument("Kernel size in TCNResidualBlock must be odd.");
  }
  unsigned int dilation = 1u;
  for (unsigned i = 0u; i < nr_residual_blocks; i++) {
    this->tcn_sequence->push_back(TCNResidualBlock(
        nr_channels, nr_channels, kernel_size, dilation, dropout));
    dilation <<= 1u;
  }
  this->register_module("tcn_sequence", this->tcn_sequence);
}
Tensor TCNFullNetwork::forward(const Tensor &x) {
  return this->tcn_sequence->forward(x);
}
} // namespace torch::nn