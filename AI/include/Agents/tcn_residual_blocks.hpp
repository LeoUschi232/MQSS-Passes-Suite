#ifndef TCN_HPP
#define TCN_HPP

// Torch includes
#include "agent_layers_and_networks.hpp"
#include "torch/torch.h"

namespace ai_pass_selector {
class TCNResidualBlock : public torch::nn::Module {
protected:
  torch::nn::WeightNormConv1d weight_norm_conv1{nullptr};
  torch::nn::WeightNormConv1d weight_norm_conv2{nullptr};
  torch::nn::Conv1d downsample{nullptr};
  torch::nn::Sequential convolutional_block{nullptr};

public:
  TCNResidualBlock(unsigned int in_channels, unsigned int out_channels,
                   unsigned int kernel_size, unsigned int dilation);
};
class TCNResidualBlockWithReLU final : public TCNResidualBlock {
  torch::nn::ReLU final_relu{nullptr};

public:
  TCNResidualBlockWithReLU(unsigned int in_channels, unsigned int out_channels,
                           unsigned int kernel_size, unsigned int dilation,
                           double dropout = 0.2);
  torch::Tensor forward(const torch::Tensor &x);
};
class TCNResidualBlockWithPReLU final : public TCNResidualBlock {
  torch::nn::PReLU final_prelu{nullptr};

public:
  TCNResidualBlockWithPReLU(unsigned int in_channels, unsigned int out_channels,
                            unsigned int kernel_size, unsigned int dilation,
                            double dropout = 0.2,
                            double final_prelu_init = 0.1);
  torch::Tensor forward(const torch::Tensor &x);
};
} // namespace ai_pass_selector

#endif // TCN_HPP
