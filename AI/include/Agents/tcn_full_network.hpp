#ifndef TCN_FULL_NETWORK_HPP
#define TCN_FULL_NETWORK_HPP

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
class TCNFullNetwork : public torch::nn::Module {
protected:
  torch::nn::Sequential network{nullptr};

public:
  explicit TCNFullNetwork(unsigned int kernel_size);
  torch::Tensor forward(const torch::Tensor &x);
};
class TCNFullNetworkWithReLU final : public TCNFullNetwork {
public:
  explicit TCNFullNetworkWithReLU(unsigned int nr_channels,
                                  unsigned int nr_residual_blocks = 12u,
                                  unsigned int kernel_size = 5u,
                                  double dropout = 0.2);
  explicit TCNFullNetworkWithReLU(
      const std::vector<unsigned int> &nr_channels_per_layer,
      unsigned int kernel_size = 5u, double dropout = 0.2);
};
class TCNFullNetworkWithPReLU final : public TCNFullNetwork {
public:
  explicit TCNFullNetworkWithPReLU(unsigned int nr_channels,
                                   unsigned int nr_residual_blocks = 12u,
                                   unsigned int kernel_size = 5u,
                                   double dropout = 0.2);
  explicit TCNFullNetworkWithPReLU(
      const std::vector<unsigned int> &nr_channels_per_layer,
      unsigned int kernel_size = 5u, double dropout = 0.2);
};

} // namespace ai_pass_selector
#endif // TCN_FULL_NETWORK_HPP
