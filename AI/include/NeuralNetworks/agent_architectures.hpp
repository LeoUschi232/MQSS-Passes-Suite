#ifndef AGENT_ARCHITECTURES_HPP
#define AGENT_ARCHITECTURES_HPP

// Torch includes
#include "torch/torch.h"

namespace ai_pass_selector {
// For TCN/Hybrid architectures the depth, and with it the maximum receptive
// field, should be dependant on the maximum number of qubits in the circuits.
// The authors of the paper on TCN recommended depths between 8 and 12.
// It makes sense make the size of the receptive field scale linearly with the
// number of qubits.
// Arbitrarily choosing a depth of 12 for pure TCN architectures for the
// MQTBench case of max_qubits=130, we have to find an 'x' such that:
// log2(x*130)=12 => x=31.5.
constexpr double TCN_QUBIT_MAGIC = 31.5;
// Arbitrarily choosing a depth of 9 for pure Hybrid architectures for the
// MQTBench case of max_qubits=130, we have to find an 'x' such that:
// log2(x*130)=9 => x=3.94.
constexpr double HYBRID_QUBIT_MAGIC = 3.93;

torch::nn::Sequential make_TCN_actor(unsigned int max_qubits,
                                     unsigned int nr_residual_blocks,
                                     unsigned int kernel_size);
torch::nn::Sequential make_TCN_critic(unsigned int max_qubits,
                                      unsigned int nr_residual_blocks,
                                      unsigned int kernel_size);
torch::nn::Sequential make_TCN_Q_estimator(unsigned int max_qubits,
                                           unsigned int nr_residual_blocks,
                                           unsigned int kernel_size);

torch::nn::Sequential make_LSTM_actor(unsigned int max_qubits,
                                      unsigned int hidden_size_multiplier,
                                      unsigned int projection_size_multiplier);
torch::nn::Sequential make_LSTM_critic(unsigned int max_qubits,
                                       unsigned int hidden_size_multiplier,
                                       unsigned int projection_size_multiplier);
torch::nn::Sequential
make_LSTM_Q_estimator(unsigned int max_qubits,
                      unsigned int hidden_size_multiplier,
                      unsigned int projection_size_multiplier);

torch::nn::Sequential
make_hybrid_actor(unsigned int max_qubits, unsigned int nr_residual_blocks,
                  unsigned int kernel_size, unsigned int hidden_size_multiplier,
                  unsigned int projection_size_multiplier);
torch::nn::Sequential
make_hybrid_critic(unsigned int max_qubits, unsigned int nr_residual_blocks,
                   unsigned int kernel_size,
                   unsigned int hidden_size_multiplier,
                   unsigned int projection_size_multiplier);
torch::nn::Sequential make_hybrid_Q_estimator(
    unsigned int max_qubits, unsigned int nr_residual_blocks,
    unsigned int kernel_size, unsigned int hidden_size_multiplier,
    unsigned int projection_size_multiplier);

} // namespace ai_pass_selector

#endif // AGENT_ARCHITECTURES_HPP
