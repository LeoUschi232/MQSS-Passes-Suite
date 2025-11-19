#ifndef TCN_NETWORK_HPP
#define TCN_NETWORK_HPP

// Torch includes
#include "torch/torch.h"

namespace torch::nn {
class TCNResidualBlock : public Module {
protected:
  Sequential convolutional_block{nullptr};
  Sequential layer_norm{nullptr};

public:
  TCNResidualBlock(unsigned int in_channels, unsigned int out_channels,
                   unsigned int kernel_size, unsigned int dilation,
                   double dropout = 0.2);
  Tensor forward(const Tensor &x);
};
class TCNFullNetwork : public Module {
protected:
  Sequential tcn_sequence{nullptr};

public:
  explicit TCNFullNetwork(unsigned int nr_channels,
                          unsigned int nr_residual_blocks,
                          unsigned int kernel_size, double dropout = 0.2);
  Tensor forward(const Tensor &x);
};
} // namespace torch::nn
#endif // TCN_NETWORK_HPP
