#include "NeuralNetworks/remapped_normalized_lstm.hpp"

// Neural Networks includes
#include "NeuralNetworks/layers_and_wrappers.hpp"

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
RemappedNormalizedLSTMImpl::RemappedNormalizedLSTMImpl(unsigned int input_size,
                                                       unsigned int hidden_size,
                                                       bool bidirectional,
                                                       unsigned int proj_size)
    : remapper_normalizer(torch::nn::Sequential(        // Input: [N, IRS]
          torch::nn::TransposeContiguous(0u, 1u),       // -> [IRS, N]
          torch::nn::Conv1d(input_size, input_size, 1), // -> [IRS, N]
          torch::nn::TransposeContiguous(0u, 1u),       // -> [N, IRS]
          torch::nn::LayerNorm(torch::nn::LayerNormOptions(
              /*normalized_shape*/ {input_size})), // -> [N, IRS]
          torch::nn::ReLU()                        // -> [N, IRS]
          )),
      my_lstm(
          0u < proj_size && proj_size < hidden_size
              ? torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
                                    .bidirectional(bidirectional)
                                    .proj_size(proj_size))
              : torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
                                    .bidirectional(bidirectional))) {
  this->register_module("remapper_normalizer", this->remapper_normalizer);
  this->register_module("my_lstm", this->my_lstm);
  this->my_lstm->flatten_parameters();
}

torch::Tensor RemappedNormalizedLSTMImpl::forward(torch::Tensor x) {
  if (x.dim() != 2 && x.dim() != 3) {
    throw std::invalid_argument(
        "FilterLSTM expects input tensor of dimension 2 or 3.");
  }
  if (x.dim() == 2) {
    x = x.unsqueeze(/*dim=*/1);
  }
  if (this->first_forward) {
    this->my_lstm->flatten_parameters();
    this->first_forward = false;
  }
  auto [y, _] = this->my_lstm->forward(this->remapper_normalizer->forward(x));
  return y.squeeze(/*dim=*/1);
}
} // namespace ai_pass_selector