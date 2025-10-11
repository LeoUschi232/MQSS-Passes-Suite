#ifndef AGENT_LAYERS_AND_NETWORKS_HPP
#define AGENT_LAYERS_AND_NETWORKS_HPP

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
  int64_t in_channels;
  int64_t out_channels;
  int64_t kernel_size;
  int64_t dilation;
  int64_t padding;
  int64_t stride = 1;
  double epsilon = 1e-12;

public:
  WeightNormConv1dImpl(int64_t in_channels, int64_t out_channels,
                       int64_t kernel_size, int64_t dilation)
      : in_channels(in_channels), out_channels(out_channels),
        kernel_size(kernel_size), dilation(dilation),
        padding(dilation * (kernel_size - 1) / 2) {
    if (kernel_size % 2 == 0) {
      throw std::invalid_argument(
          "Kernel size in WeightNormConv1dImpl must be odd.");
    }
    std::vector weight_shape = {out_channels, in_channels, kernel_size};
    weight_v = register_parameter("weight_v", torch::empty(weight_shape));
    // Syntax of norm(2,{1,2},true)
    // p=2: L2-Norm
    // dim={1,2}: Take the norm over dimensionss 1 and 2, so over C_in and K
    // leaving dimension 0, so C_out alone.
    // keepdims=true: Keep reduced dimensions with size 1, so g can broadcast
    // back to [C_out, C_in, K].
    weight_g =
        register_parameter("weight_g", weight_v.norm(/*p=*/2, /*dim=*/{1, 2},
                                                     /*keepdim=*/true));
  }

  Tensor forward(const Tensor &input) {
    return conv1d(input,
                  weight_g * weight_v /
                      (weight_v.norm(/*p=*/2, /*dim=*/{1, 2},
                                     /*keepdim=*/true) +
                       epsilon),
                  Tensor(), {stride}, {padding}, {dilation});
  }
};

TORCH_MODULE(WeightNormConv1d);

/**
 * Nonlinearity modReLU from the paper Tunable Efficient Unitary Neural Networks
 * (EUNN) and their application to RNNs.
 */
class ModReLUImpl final : public Module {
  Tensor bias;

public:
  explicit ModReLUImpl(int64_t num_features) {
    bias = register_parameter("bias", zeros({num_features}));
  }
  Tensor forward(const Tensor &z) const {
    return sign(z) * relu(abs(z) + bias);
  }
};
TORCH_MODULE(ModReLU);

/**
 * Instruction tensor will have shape [N, IRS]
 * N = Nr of instructions in the quantum circuit
 * IRS = Instruction Representation Size
 * The convolutional layer expects input of shape [nr_channels, Length]
 * This means the instruction tensor [N, IRS] must be transposed to [IRS, N]
 * @param dim0
 * @param dim1
 * @return
 */
inline Functional TransposeContiguous(int64_t dim0, int64_t dim1) {
  return Functional([dim0, dim1](const Tensor &x) {
    return x.transpose(dim0, dim1).contiguous();
  });
}

/**
 * Instruction tensor will have shape [N, IRS]
 * N = Nr of instructions in the quantum circuit
 * IRS = Instruction Representation Size
 * The convolutional layer expects input of shape [nr_channels, Length]
 * This means the instruction tensor [N, IRS] must be transposed to [IRS, N]
 * @param dim0
 * @param dim1
 * @return
 */
inline Functional Transpose(int64_t dim0, int64_t dim1) {
  return Functional(
      [dim0, dim1](const Tensor &x) { return x.transpose(dim0, dim1); });
}
} // namespace torch::nn

#endif // AGENT_LAYERS_AND_NETWORKS_HPP
