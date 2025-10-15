#ifndef WEIGHT_NORM_CONV1D_HPP
#define WEIGHT_NORM_CONV1D_HPP

// Torch includes
#include "torch/torch.h"

namespace torch::nn {
/**
 * Weight Normalized 1D Convolutional Layer.
 * Specially made for the TCN Residual Blocks.
 */
class WeightNormConv1dImpl final : public Module {
  /// WeightNorm Parameters
  Tensor weight_v;
  Tensor weight_g;

  /// Conv1d Parameters
  int32_t in_channels;
  int32_t out_channels;
  int32_t kernel_size;
  int32_t dilation;
  int32_t padding;
  Tensor bias;

  // Constant parameters
  const int32_t stride = 1;
  const int32_t normL2 = 2;
  const std::array<int64_t, 2> normDims = {1, 2};
  const bool keepDims = true;

  /// Extra Parameters
  const double epsilon = 1e-8;

public:
  WeightNormConv1dImpl(int32_t in_channels, int32_t out_channels,
                       int32_t kernel_size, int32_t dilation);

  void init_weights() const;

  Tensor forward(const Tensor &input);
};

TORCH_MODULE(WeightNormConv1d);
} // namespace torch::nn
#endif // WEIGHT_NORM_CONV1D_HPP
