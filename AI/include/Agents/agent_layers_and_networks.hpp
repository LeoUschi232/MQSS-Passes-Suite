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
                       int32_t kernel_size, int32_t dilation)
      : in_channels(in_channels), out_channels(out_channels),
        kernel_size(kernel_size), dilation(dilation),
        padding(dilation * (kernel_size - 1) / 2) {
    if (kernel_size % 2 == 0) {
      throw std::invalid_argument(
          "Kernel size in WeightNormConv1dImpl must be odd.");
    }
    std::vector<int64_t> weight_shape = {out_channels, in_channels,
                                         kernel_size};
    this->weight_v = register_parameter("weight_v", torch::empty(weight_shape));
    this->weight_g = register_parameter("weight_g", ones({out_channels, 1, 1}));
    this->bias = register_parameter("bias", zeros({out_channels}));
    this->init_weights();
  }

  void init_weights() const {
    NoGradGuard _;
    (void)this->weight_v.normal_(0, 0.01);
    (void)this->weight_g.fill_(1);
  }

  Tensor forward(const Tensor &input) {
    // Syntax of norm(2,{1,2},true)
    // p=2: L2-Norm
    // dim={1,2}: Take the norm over dimensionss 1 and 2, so over C_in and K
    // leaving dimension 0, so C_out alone.
    // keepdims=true: Keep reduced dimensions with size 1, so g can broadcast
    // back to [C_out, C_in, K].
    auto the_norm = weight_v.norm(/*p=*/this->normL2, /*dim=*/this->normDims,
                                  /*keepdim=*/this->keepDims);
    return conv1d(input, weight_g * weight_v / (the_norm + epsilon), this->bias,
                  this->stride, this->padding, this->dilation);
  }
};

TORCH_MODULE(WeightNormConv1d);

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
inline Functional TransposeContiguous(int32_t dim0, int32_t dim1) {
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
inline Functional Transpose(int32_t dim0, int32_t dim1) {
  return Functional(
      [dim0, dim1](const Tensor &x) { return x.transpose(dim0, dim1); });
}

/**
 * Check for NaN and Inf values in the tensor.
 * If any are found, print the stage name, min and max values.
 * @param stage_name Name of the stage to identify where the values might be inf
 * or nan.
 * @return
 */
inline Functional FiniteCheck(std::string stage_name) {
  return Functional([name = std::move(stage_name)](const Tensor &tensor) {
    if (const bool has_inf = tensor.isinf().any().item<bool>(),
        has_nan = tensor.isnan().any().item<bool>();
        has_nan || has_inf) {
      const double min_val = tensor.amin().item<double>();
      const double max_val = tensor.amax().item<double>();
      std::cerr << "[FiniteCheck] " << name << " nan=" << has_nan
                << " inf=" << has_inf << " min=" << min_val
                << " max=" << max_val << "\n";
    }
    return tensor;
  });
}

/**
 *
 * @param stage_name
 * @return
 */
inline Functional ShapeProbe(std::string stage_name) {
  return Functional([name = std::move(stage_name)](const Tensor &x) {
    std::ostringstream oss;
    oss << "[ShapeProbe] " << name << " sizes=[";
    for (size_t i = 0; i < x.sizes().size(); ++i) {
      oss << x.sizes()[i] << (i + 1 < x.sizes().size() ? "," : "");
    }
    oss << "] min=" << x.amin().item<double>()
       << " max=" << x.amax().item<double>()
       << " finite=" << x.isfinite().all().item<bool>() << "\n";
    std::cerr << oss.str();
    return x;
  });
}

} // namespace torch::nn

#endif // AGENT_LAYERS_AND_NETWORKS_HPP
