#ifndef TCN_RESIDUAL_BLOCK_HPP
#define TCN_RESIDUAL_BLOCK_HPP

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
class TCNResidualBlock : public torch::nn::Module {
protected:
  torch::nn::Sequential convolutional_block{nullptr};
  torch::nn::Sequential layer_norm{nullptr};

public:
  TCNResidualBlock(unsigned int in_channels, unsigned int out_channels,
                   unsigned int kernel_size, unsigned int dilation,
                   double dropout = 0.2);
  torch::Tensor forward(const torch::Tensor &x);
};
} // namespace ai_pass_selector
#endif // TCN_RESIDUAL_BLOCK_HPP
