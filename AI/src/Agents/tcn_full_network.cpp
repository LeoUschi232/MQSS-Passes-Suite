#include "Agents/tcn_full_network.hpp"

// Agents includes
#include "Agents/tcn_residual_blocks.hpp"

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
TCNFullNetwork::TCNFullNetwork(unsigned int kernel_size) {
  if (kernel_size % 2 == 0) {
    throw std::invalid_argument("Kernel size in TCNResidualBlock must be odd.");
  }
}
torch::Tensor TCNFullNetwork::forward(const torch::Tensor &x) {
  if (!this->network) {
    throw std::runtime_error("TCNFullNetwork network not initialized.");
  }
  return this->network->forward(x);
}

TCNFullNetworkWithReLU::TCNFullNetworkWithReLU(unsigned int nr_channels,
                                               unsigned int nr_residual_blocks,
                                               unsigned int kernel_size,
                                               double dropout)
    : TCNFullNetwork(kernel_size) {
  this->network = torch::nn::Sequential();
  unsigned int dilation = 1u;
  for (unsigned i = 0u; i < nr_residual_blocks; i++) {
    this->network->push_back(TCNResidualBlockWithReLU(
        nr_channels, nr_channels, kernel_size, dilation, dropout));
    dilation <<= 1u;
  }
  this->register_module("network", this->network);
}

TCNFullNetworkWithReLU::TCNFullNetworkWithReLU(
    const std::vector<unsigned int> &nr_channels_per_layer,
    unsigned int kernel_size, double dropout)
    : TCNFullNetwork(kernel_size) {
  this->network = torch::nn::Sequential();
  unsigned int dilation = 1u;
  for (unsigned int i = 1u; i < nr_channels_per_layer.size(); i++) {
    this->network->push_back(TCNResidualBlockWithReLU(
        nr_channels_per_layer[i - 1u], nr_channels_per_layer[i], kernel_size,
        dilation, dropout));
    dilation <<= 1u;
  }
  this->register_module("network", this->network);
}

TCNFullNetworkWithPReLU::TCNFullNetworkWithPReLU(
    unsigned int nr_channels, unsigned int nr_residual_blocks,
    unsigned int kernel_size, double prelu_init, double dropout)
    : TCNFullNetwork(kernel_size) {
  this->network = torch::nn::Sequential();
  unsigned int dilation = 1u;
  for (unsigned i = 0u; i < nr_residual_blocks; i++) {
    this->network->push_back(TCNResidualBlockWithPReLU(
        nr_channels, nr_channels, kernel_size, dilation, prelu_init, dropout));
    dilation <<= 1u;
  }
  this->register_module("network", this->network);
}

TCNFullNetworkWithPReLU::TCNFullNetworkWithPReLU(
    const std::vector<unsigned int> &nr_channels_per_layer,
    unsigned int kernel_size, double prelu_init, double dropout)
    : TCNFullNetwork(kernel_size) {
  this->network = torch::nn::Sequential();
  unsigned int dilation = 1u;
  for (unsigned int i = 1u; i < nr_channels_per_layer.size(); i++) {
    this->network->push_back(TCNResidualBlockWithPReLU(
        nr_channels_per_layer[i - 1u], nr_channels_per_layer[i], kernel_size,
        dilation, prelu_init, dropout));
    dilation <<= 1u;
  }
  this->register_module("network", this->network);
}
} // namespace ai_pass_selector