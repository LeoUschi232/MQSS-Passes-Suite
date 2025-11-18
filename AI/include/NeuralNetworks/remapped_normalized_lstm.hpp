#ifndef REMAPPED_NORMALIZED_LSTM_HPP
#define REMAPPED_NORMALIZED_LSTM_HPP

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
/**
 * The LSTM in libtorch returns a std::tuple<Tensor, std::tuple<Tensor,
 * Tensor>> where only the first tensor is the actual output of the LSTM. The
 * other two tensors are the hidden and cell states, which we do not need.
 * This functional extracts only the first tensor from the tuple.
 * This function MUST be placed right after an LSTM in a nn::Sequential.
 * @return The actual output tensor of the LSTM.
 */
class SingleOutputLSTMImpl final : public torch::nn::Module {
  torch::nn::LSTM my_lstm{nullptr};
  bool first_forward = true;

public:
  SingleOutputLSTMImpl(unsigned int input_size, unsigned int hidden_size,
                       unsigned int proj_size = 0);
  torch::Tensor forward(const torch::Tensor &x);
};
TORCH_MODULE(SingleOutputLSTM);

} // namespace ai_pass_selector

#endif // REMAPPED_NORMALIZED_LSTM_HPP
