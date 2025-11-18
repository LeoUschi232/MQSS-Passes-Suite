#include "NeuralNetworks/tcn_full_network.hpp"

// Agents includes
#include "NeuralNetworks/tcn_residual_blocks.hpp"

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
TCNFullNetwork::TCNFullNetwork(unsigned int nr_channels,
                               unsigned int nr_residual_blocks,
                               unsigned int kernel_size, double dropout)
    : tcn_sequence(torch::nn::Sequential()) {
  if (kernel_size % 2 == 0) {
    throw std::invalid_argument("Kernel size in TCNResidualBlock must be odd.");
  }
  unsigned int dilation = 1u;
  for (unsigned i = 0u; i < nr_residual_blocks; i++) {
    this->tcn_sequence->push_back(TCNResidualBlock(
        /*in_channels=*/nr_channels, /*out_channels=*/nr_channels, kernel_size,
        dilation, dropout));
    dilation <<= 1u;
  }
  this->register_module("tcn_sequence", this->tcn_sequence);
}
torch::Tensor TCNFullNetwork::forward(const torch::Tensor &x) {
  return this->tcn_sequence->forward(x);
}
} // namespace ai_pass_selector