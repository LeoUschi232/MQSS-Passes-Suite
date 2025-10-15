#include "NeuralNetworks/weight_norm_conv1d.hpp"

// Torch includes
#include "torch/torch.h"

namespace torch::nn {
WeightNormConv1dImpl::WeightNormConv1dImpl(int32_t in_channels, int32_t out_channels,
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
  return conv1d(input, this->weight_g * (this->weight_v / the_norm),
                this->bias, this->stride, this->padding, this->dilation);
}

} // namespace torch::nn