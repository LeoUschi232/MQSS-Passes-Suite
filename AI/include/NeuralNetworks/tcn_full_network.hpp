#ifndef TCN_FULL_NETWORK_HPP
#define TCN_FULL_NETWORK_HPP

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
class TCNFullNetwork : public torch::nn::Module {
protected:
  torch::nn::Sequential tcn_sequence{nullptr};

public:
  explicit TCNFullNetwork(unsigned int nr_channels,
                          unsigned int nr_residual_blocks,
                          unsigned int kernel_size, double dropout = 0.2);
  torch::Tensor forward(const torch::Tensor &x);
};
} // namespace ai_pass_selector
#endif // TCN_FULL_NETWORK_HPP
