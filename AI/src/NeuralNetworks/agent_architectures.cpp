#include "NeuralNetworks/agent_architectures.hpp"

// Neural Networks includes
#include "NeuralNetworks/layers_and_wrappers.hpp"
#include "NeuralNetworks/remapped_normalized_lstm.hpp"
#include "NeuralNetworks/tcn_full_network.hpp"

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/passes_utils.hpp"

namespace ai_pass_selector {
torch::nn::Sequential
make_TCN_actor(unsigned int max_qubits, unsigned int nr_residual_blocks,
               unsigned int kernel_size,
               const std::optional<double> &optional_prelu_init) {
  // Input shape: [N, IRS]
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  return torch::nn::Sequential(               // Input: [N, IRS]
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      optional_prelu_init.has_value()
          ? torch::nn::AnyModule(
                TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                        optional_prelu_init.value()))
          : torch::nn::AnyModule(
                TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                       kernel_size)), // -> [IRS, N]
      torch::nn::AdaptiveAvgPool1d(1u),               // -> [IRS, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [IRS]
      torch::nn::Linear(IRS, NR_PASSES),                     // -> [NR_PASSES]
      torch::nn::Softmax(/*dim=*/0u)                         // -> [NR_PASSES]
  );
}
torch::nn::Sequential
make_TCN_critic(unsigned int max_qubits, unsigned int nr_residual_blocks,
                unsigned int kernel_size,
                const std::optional<double> &optional_prelu_init) {
  // Input shape: [N, IRS]
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  return torch::nn::Sequential(               // Input: [N, IRS]
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      optional_prelu_init.has_value()
          ? torch::nn::AnyModule(
                TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                        optional_prelu_init.value()))
          : torch::nn::AnyModule(
                TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                       kernel_size)), // -> [IRS, N]
      torch::nn::AdaptiveAvgPool1d(1u),               // -> [IRS, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [IRS]
      torch::nn::Linear(IRS, 1),                             // -> [1]
      torch::nn::Squeeze(/*dim=*/0)                          // -> []
  );
}
torch::nn::Sequential
make_TCN_Q_estimator(unsigned int max_qubits, unsigned int nr_residual_blocks,
                     unsigned int kernel_size,
                     const std::optional<double> &optional_prelu_init) {
  // Input shape: [N, IRS]
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  return torch::nn::Sequential(               // Input: [N, IRS]
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      optional_prelu_init.has_value()
          ? torch::nn::AnyModule(
                TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                        optional_prelu_init.value()))
          : torch::nn::AnyModule(
                TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                       kernel_size)), // -> [IRS, N]
      torch::nn::AdaptiveAvgPool1d(1u),               // -> [IRS, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [IRS]
      torch::nn::Linear(IRS, NR_PASSES)                      // -> [NR_PASSES]
  );
}
torch::nn::Sequential make_LSTM_actor(unsigned int max_qubits,
                                      unsigned int hidden_size_multiplier,
                                      unsigned int projection_size_multiplier) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  const unsigned int H = hidden_size_multiplier * IRS;
  const unsigned int P = projection_size_multiplier * IRS;
  return torch::nn::Sequential( // Input: [N, IRS]
      RemappedNormalizedLSTM(/*input_size=*/IRS, /*hidden_size=*/H,
                             /*bidirectional=*/true,
                             /*proj_size=*/P), // -> [N, 2*P]
      torch::nn::LayerNorm(torch::nn::LayerNormOptions(
          /*normalized_shape=*/{2 * P})),     // -> [N, 2*P]
      torch::nn::TransposeContiguous(0u, 1u), // -> [2*P, N]
      torch::nn::AdaptiveAvgPool1d(1u),       // -> [2*P, 1]
      torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(0)), // -> [2*P]
      torch::nn::Linear(2 * P, NR_PASSES), // -> [NR_PASSES]
      torch::nn::Softmax(/*dim=*/0u)       // -> [NR_PASSES]
  );
}
torch::nn::Sequential
make_LSTM_critic(unsigned int max_qubits, unsigned int hidden_size_multiplier,
                 unsigned int projection_size_multiplier) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  const unsigned int H = hidden_size_multiplier * IRS;
  const unsigned int P = projection_size_multiplier * IRS;
  return torch::nn::Sequential( // Input: [N, IRS]
      RemappedNormalizedLSTM(/*input_size=*/IRS, /*hidden_size=*/H,
                             /*bidirectional=*/true,
                             /*proj_size=*/P), // -> [N, 2*P]
      torch::nn::LayerNorm(torch::nn::LayerNormOptions(
          /*normalized_shape=*/{2 * P})),     // -> [N, 2*P]
      torch::nn::TransposeContiguous(0u, 1u), // -> [2*P, N]
      torch::nn::AdaptiveAvgPool1d(1u),       // -> [2*P, 1]
      torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(0)), // -> [2*P]
      torch::nn::Linear(2 * P, 1),                                  // -> [1]
      torch::nn::Squeeze(/*dim=*/0)                                 // -> []
  );
}
torch::nn::Sequential
make_LSTM_Q_estimator(unsigned int max_qubits,
                      unsigned int hidden_size_multiplier,
                      unsigned int projection_size_multiplier) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  const unsigned int H = hidden_size_multiplier * IRS;
  const unsigned int P = projection_size_multiplier * IRS;
  return torch::nn::Sequential( // Input: [N, IRS]
      RemappedNormalizedLSTM(/*input_size=*/IRS, /*hidden_size=*/H,
                             /*bidirectional=*/true,
                             /*proj_size=*/P), // -> [N, 2*P]
      torch::nn::LayerNorm(torch::nn::LayerNormOptions(
          /*normalized_shape=*/{2 * P})),     // -> [N, 2*P]
      torch::nn::TransposeContiguous(0u, 1u), // -> [2*P, N]
      torch::nn::AdaptiveAvgPool1d(1u),       // -> [2*P, 1]
      torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(0)), // -> [2*P]
      torch::nn::Linear(2 * P, NR_PASSES) // -> [NR_PASSES]
  );
}

torch::nn::Sequential
make_hybrid_actor(unsigned int max_qubits, unsigned int nr_residual_blocks,
                  unsigned int kernel_size, unsigned int hidden_size_multiplier,
                  unsigned int projection_size_multiplier,
                  const std::optional<double> &optional_prelu_init) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  const unsigned int H = hidden_size_multiplier * IRS;
  const unsigned int P = projection_size_multiplier * IRS;
  return torch::nn::Sequential(               // Input: [N, IRS]
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      optional_prelu_init.has_value()
          ? torch::nn::AnyModule(
                TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                        optional_prelu_init.value()))
          : torch::nn::AnyModule(
                TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                       kernel_size)), // -> [IRS, N]
      torch::nn::TransposeContiguous(0u, 1u),         // -> [N, IRS]
      RemappedNormalizedLSTM(/*input_size=*/IRS, /*hidden_size=*/H,
                             /*bidirectional=*/true,
                             /*proj_size=*/P), // -> [N, 2*P]
      torch::nn::LayerNorm(torch::nn::LayerNormOptions(
          /*normalized_shape=*/{2 * P})),     // -> [N, 2*P]
      torch::nn::TransposeContiguous(0u, 1u), // -> [2*P, N]
      torch::nn::AdaptiveAvgPool1d(1u),       // -> [2*P, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [2*P]
      torch::nn::Linear(2 * P, NR_PASSES),                   // -> [NR_PASSES]
      torch::nn::Softmax(/*dim=*/0u)                         // -> [NR_PASSES]
  );
}
torch::nn::Sequential
make_hybrid_critic(unsigned int max_qubits, unsigned int nr_residual_blocks,
                   unsigned int kernel_size,
                   unsigned int hidden_size_multiplier,
                   unsigned int projection_size_multiplier,
                   const std::optional<double> &optional_prelu_init) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  const unsigned int H = hidden_size_multiplier * IRS;
  const unsigned int P = projection_size_multiplier * IRS;
  return torch::nn::Sequential(               // Input: [N, IRS]
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      optional_prelu_init.has_value()
          ? torch::nn::AnyModule(
                TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                        optional_prelu_init.value()))
          : torch::nn::AnyModule(
                TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                       kernel_size)), // -> [IRS, N]
      torch::nn::TransposeContiguous(0u, 1u),         // -> [N, IRS]
      RemappedNormalizedLSTM(/*input_size=*/IRS, /*hidden_size=*/H,
                             /*bidirectional=*/true,
                             /*proj_size=*/P), // -> [N, 2*P]
      torch::nn::LayerNorm(torch::nn::LayerNormOptions(
          /*normalized_shape=*/{2 * P})),     // -> [N, 2*P]
      torch::nn::TransposeContiguous(0u, 1u), // -> [2*P, N]
      torch::nn::AdaptiveAvgPool1d(1u),       // -> [2*P, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [2*P]
      torch::nn::Linear(2 * P, 1),                           // -> [1]
      torch::nn::Squeeze(/*dim=*/0)                          // -> []
  );
}
torch::nn::Sequential make_hybrid_Q_estimator(
    unsigned int max_qubits, unsigned int nr_residual_blocks,
    unsigned int kernel_size, unsigned int hidden_size_multiplier,
    unsigned int projection_size_multiplier,
    const std::optional<double> &optional_prelu_init) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  const unsigned int H = hidden_size_multiplier * IRS;
  const unsigned int P = projection_size_multiplier * IRS;
  return torch::nn::Sequential(               // Input: [N, IRS]
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      optional_prelu_init.has_value()
          ? torch::nn::AnyModule(
                TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                        optional_prelu_init.value()))
          : torch::nn::AnyModule(
                TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                       kernel_size)), // -> [IRS, N]
      torch::nn::TransposeContiguous(0u, 1u),         // -> [N, IRS]
      RemappedNormalizedLSTM(/*input_size=*/IRS, /*hidden_size=*/H,
                             /*bidirectional=*/true,
                             /*proj_size=*/P), // -> [N, 2*P]
      torch::nn::LayerNorm(torch::nn::LayerNormOptions(
          /*normalized_shape=*/{2 * P})),     // -> [N, 2*P]
      torch::nn::TransposeContiguous(0u, 1u), // -> [2*P, N]
      torch::nn::AdaptiveAvgPool1d(1u),       // -> [2*P, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [2*P]
      torch::nn::Linear(2 * P, NR_PASSES)                    // -> [NR_PASSES]
  );
}

} // namespace ai_pass_selector
