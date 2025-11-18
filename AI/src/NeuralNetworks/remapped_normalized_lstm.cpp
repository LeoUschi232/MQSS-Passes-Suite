#include "NeuralNetworks/remapped_normalized_lstm.hpp"

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
SingleOutputLSTMImpl::SingleOutputLSTMImpl(unsigned int input_size,
                                           unsigned int hidden_size,
                                           unsigned int proj_size)
    : my_lstm( // Input: [N, C_in]
          0u < proj_size && proj_size < hidden_size
              ? torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
                                    .bidirectional(true)
                                    .proj_size(proj_size)) // -> [N, 2*P]
              : torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
                                    .bidirectional(true)) // -> [N, 2*H]
      ) {
  this->register_module("my_lstm", this->my_lstm);
  this->my_lstm->flatten_parameters();
}

torch::Tensor SingleOutputLSTMImpl::forward(const torch::Tensor &x) {
  if (this->first_forward) {
    this->my_lstm->flatten_parameters();
    this->first_forward = false;
  }
  auto [y, _] = this->my_lstm->forward(x.unsqueeze(1));
  return y.squeeze(1);
}
} // namespace ai_pass_selector