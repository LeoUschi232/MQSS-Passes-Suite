#ifndef LAYERS_AND_WRAPPERS_HPP
#define LAYERS_AND_WRAPPERS_HPP

// Torch includes
#include "torch/torch.h"

namespace torch::nn {
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
  const double epsilon = 1e-12;

public:
  WeightNormConv1dImpl(int32_t in_channels, int32_t out_channels,
                       int32_t kernel_size, int32_t dilation);

  void init_weights() const;
  Tensor forward(const Tensor &input);
};

TORCH_MODULE(WeightNormConv1d);

/**
 * The LSTM in libtorch returns a std::tuple<Tensor, std::tuple<Tensor,
 * Tensor>> where only the first tensor is the actual output of the LSTM. The
 * other two tensors are the hidden and cell states, which we do not need.
 * This functional extracts only the first tensor from the tuple.
 * This function MUST be placed right after an LSTM in a nn::Sequential.
 * @return The actual output tensor of the LSTM.
 */
class FilterLSTMImpl final : public Module {
  LSTM my_lstm{nullptr};
  bool first_forward = true;

public:
  FilterLSTMImpl(unsigned int input_size, unsigned int hidden_size,
                 bool bidirectional, unsigned int proj_size = 0);
  Tensor forward(Tensor x);
};
TORCH_MODULE(FilterLSTM);

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
Functional TransposeContiguous(int32_t dim0, int32_t dim1);

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
Functional Transpose(int32_t dim0, int32_t dim1);

/**
 *
 * @param dim
 * @return
 */
Functional Squeeze(int32_t dim);

/**
 * Check for NaN and Inf values in the tensor.
 * If any are found, print the stage name, min and max values.
 * @param stage_name Name of the stage to identify where the values might be inf
 * or nan.
 * @return
 */
Functional FiniteCheck(std::string stage_name);

/**
 * Check the shape of the tensor and print it.
 * @param stage_name
 * @return
 */
Functional ShapeProbe(std::string stage_name);

/**
 *
 * @param tensor
 * @param precision
 * @return
 */
std::string tensor_to_string(const torch::Tensor &tensor, int precision = 6);

/**
 * @param precision Number of decimal places to print.
 * @return
 */
Functional PrintTensor(unsigned int precision = 3);
} // namespace torch::nn
#endif // LAYERS_AND_WRAPPERS_HPP
