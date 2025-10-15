#include "NeuralNetworks/layers_and_wrappers.hpp"

// Torch includes
#include "torch/torch.h"

namespace torch::nn {
WeightNormConv1dImpl::WeightNormConv1dImpl(int32_t in_channels,
                                           int32_t out_channels,
                                           int32_t kernel_size,
                                           int32_t dilation)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), dilation(dilation),
      padding(dilation * (kernel_size - 1) / 2) {
  if (kernel_size % 2 == 0) {
    throw std::invalid_argument(
        "Kernel size in WeightNormConv1dImpl must be odd.");
  }
  std::vector<int64_t> weight_shape = {out_channels, in_channels, kernel_size};
  this->weight_v = register_parameter("weight_v", torch::empty(weight_shape));
  this->weight_g = register_parameter("weight_g", ones({out_channels, 1, 1}));
  this->bias = register_parameter("bias", zeros({out_channels}));
  this->init_weights();
}

void WeightNormConv1dImpl::init_weights() const {
  NoGradGuard _;
  (void)this->weight_v.normal_(0, 0.01);
  (void)this->weight_g.fill_(1);
  (void)this->bias.zero_();
}

Tensor WeightNormConv1dImpl::forward(const Tensor &input) {
  // Syntax of norm(2,{1,2},true)
  // p=2: L2-Norm
  // dim={1,2}: Take the norm over dimensionss 1 and 2, so over C_in and K
  // leaving dimension 0, so C_out alone.
  // keepdims=true: Keep reduced dimensions with size 1, so g can broadcast
  // back to [C_out, C_in, K].
  auto the_norm = this->weight_v
                      .norm(/*p=*/this->normL2, /*dim=*/this->normDims,
                            /*keepdim=*/this->keepDims)
                      .clamp_min(this->epsilon);
  // Autograd will backprop through the normalization and automatically
  // produce the gradients for g and v exactly.
  return conv1d(input, this->weight_g * (this->weight_v / the_norm), this->bias,
                this->stride, this->padding, this->dilation);
}

Tensor FilterLSTMImpl::forward(
    const std::tuple<Tensor, std::tuple<Tensor, Tensor>> &lstm_output) {
  return std::get<0>(lstm_output);
}

Functional TransposeContiguous(int32_t dim0, int32_t dim1) {
  return Functional([dim0, dim1](const Tensor &x) {
    return x.transpose(dim0, dim1).contiguous();
  });
}

Functional Transpose(int32_t dim0, int32_t dim1) {
  return Functional(
      [dim0, dim1](const Tensor &x) { return x.transpose(dim0, dim1); });
}

Functional FiniteCheck(std::string stage_name) {
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

Functional ShapeProbe(std::string stage_name) {
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