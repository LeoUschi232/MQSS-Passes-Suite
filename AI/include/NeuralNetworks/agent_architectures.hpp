#ifndef AGENT_ARCHITECTURES_HPP
#define AGENT_ARCHITECTURES_HPP

// Torch includes
#include "torch/torch.h"

// Standard library includes
#include <optional>

namespace ai_pass_selector {
torch::nn::Sequential
make_TCN_actor(unsigned int max_qubits, unsigned int nr_residual_blocks,
               unsigned int kernel_size,
               const std::optional<double> &optional_prelu_init = std::nullopt);
torch::nn::Sequential make_TCN_critic(
    unsigned int max_qubits, unsigned int nr_residual_blocks,
    unsigned int kernel_size,
    const std::optional<double> &optional_prelu_init = std::nullopt);

torch::nn::Sequential make_LSTM_actor(unsigned int max_qubits,
                                      unsigned int hidden_size_multiplier,
                                      unsigned int projection_size_multiplier);
torch::nn::Sequential make_LSTM_critic(unsigned int max_qubits,
                                       unsigned int hidden_size_multiplier,
                                       unsigned int projection_size_multiplier);

torch::nn::Sequential make_HYBRID_actor(
    unsigned int max_qubits, unsigned int nr_residual_blocks,
    unsigned int kernel_size, unsigned int hidden_size_multiplier,
    unsigned int projection_size_multiplier,
    const std::optional<double> &optional_prelu_init = std::nullopt);
torch::nn::Sequential make_HYBRID_critic(
    unsigned int max_qubits, unsigned int nr_residual_blocks,
    unsigned int kernel_size, unsigned int hidden_size_multiplier,
    unsigned int projection_size_multiplier,
    const std::optional<double> &optional_prelu_init = std::nullopt);

} // namespace ai_pass_selector

#endif // AGENT_ARCHITECTURES_HPP
