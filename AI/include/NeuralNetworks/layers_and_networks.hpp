#ifndef LAYERS_AND_NETWORKS_HPP
#define LAYERS_AND_NETWORKS_HPP

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {

torch::nn::Sequential make_TCN_actor(unsigned int max_qubits);
torch::nn::Sequential make_TCN_critic(unsigned int max_qubits);
torch::nn::Sequential make_LSTM_actor(unsigned int max_qubits);
torch::nn::Sequential make_LSTM_critic(unsigned int max_qubits);
torch::nn::Sequential make_HYBRID_actor(unsigned int max_qubits);
torch::nn::Sequential make_HYBRID_critic(unsigned int max_qubits);
} // namespace ai_pass_selector

namespace torch::nn {
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
 * Check for NaN and Inf values in the tensor.
 * If any are found, print the stage name, min and max values.
 * @param stage_name Name of the stage to identify where the values might be inf
 * or nan.
 * @return
 */
Functional FiniteCheck(std::string stage_name);

/**
 *
 * @param stage_name
 * @return
 */
Functional ShapeProbe(std::string stage_name);
} // namespace torch::nn

#endif // LAYERS_AND_NETWORKS_HPP
