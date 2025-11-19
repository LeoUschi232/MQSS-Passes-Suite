#ifndef SINGLE_OUTPUT_LSTM_HPP
#define SINGLE_OUTPUT_LSTM_HPP

// Torch includes
#include "torch/torch.h"

namespace torch::nn {
/**
 * The LSTM in libtorch returns a std::tuple<Tensor, std::tuple<Tensor,
 * Tensor>> where only the first tensor is the actual output of the LSTM. The
 * other two tensors are the hidden and cell states, which we do not need.
 * This functional extracts only the first tensor from the tuple.
 * This function MUST be placed right after an LSTM in a nn::Sequential.
 * @return The actual output tensor of the LSTM.
 */
class SingleOutputLSTMImpl final : public Module {
  LSTM my_lstm{nullptr};
  bool first_forward = true;

public:
  SingleOutputLSTMImpl(unsigned int input_size, unsigned int hidden_size,
                       unsigned int proj_size = 0);
  Tensor forward(const Tensor &x);
};
TORCH_MODULE(SingleOutputLSTM);

} // namespace torch::nn

#endif // SINGLE_OUTPUT_LSTM_HPP
