#include "NeuralNetworks/agent_architectures.hpp"

// Neural-Networks includes
#include "NeuralNetworks/layers_and_wrappers.hpp"
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
  auto actor = torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u) // -> [IRS, N]
  );
  if (optional_prelu_init.has_value()) {
    actor->push_back(
        TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                optional_prelu_init.value())); // -> [IRS, N]
  } else {
    actor->push_back(TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                            kernel_size)); // -> [IRS, N]
  }
  actor->push_back(torch::nn::AdaptiveAvgPool1d(1u)); // -> [IRS, 1]
  actor->push_back(torch::nn::Flatten(
      torch::nn::FlattenOptions().start_dim(/*dim=*/0))); // -> [IRS]
  actor->push_back(torch::nn::Linear(IRS, NR_PASSES));    // -> [NR_PASSES]
  actor->push_back(torch::nn::Softmax(/*dim=*/0u));       // -> [NR_PASSES]
  return actor;
}
torch::nn::Sequential
make_TCN_critic(unsigned int max_qubits, unsigned int nr_residual_blocks,
                unsigned int kernel_size,
                const std::optional<double> &optional_prelu_init) {
  // Input shape: [N, IRS]
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  auto critic = torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u) // -> [IRS, N]
  );
  if (optional_prelu_init.has_value()) {
    critic->push_back(
        TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                optional_prelu_init.value())); // -> [IRS, N]
  } else {
    critic->push_back(TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                             kernel_size)); // -> [IRS, N]
  }
  critic->push_back(torch::nn::AdaptiveAvgPool1d(1u)); // -> [IRS, 1]
  critic->push_back(torch::nn::Flatten(
      torch::nn::FlattenOptions().start_dim(/*dim=*/0))); // -> [IRS]
  critic->push_back(torch::nn::Linear(IRS, 1));           // -> [1]
  critic->push_back(torch::nn::Squeeze(/*dim=*/0));       // -> []
  return critic;
}
torch::nn::Sequential make_LSTM_actor(unsigned int max_qubits,
                                      unsigned int hidden_size_multiplier,
                                      unsigned int projection_size_multiplier) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  const unsigned int H = hidden_size_multiplier * IRS;
  const unsigned int P = projection_size_multiplier * IRS;
  auto actor = torch::nn::Sequential();
  if (H == P) {
    actor->push_back(torch::nn::LSTM(
        torch::nn::LSTMOptions(/*input_size=*/IRS, /*hidden_size=*/H)
            .bidirectional(true))); // -> [N, (2*P, (h, c))]
  } else {
    actor->push_back(torch::nn::LSTM(
        torch::nn::LSTMOptions(/*input_size=*/IRS, /*hidden_size=*/H)
            .bidirectional(true)
            .proj_size(P))); // -> [N, (2*P, (h, c))]
  }
  actor->push_back(torch::nn::FilterLSTM());                // -> [N, 2*P]
  actor->push_back(torch::nn::TransposeContiguous(0u, 1u)); // -> [2*P, N]
  actor->push_back(torch::nn::AdaptiveAvgPool1d(1u));       // -> [2*P, 1]
  actor->push_back(torch::nn::Flatten(
      torch::nn::FlattenOptions().start_dim(/*dim=*/0))); // -> [2*P]
  actor->push_back(torch::nn::Linear(2 * P, NR_PASSES));  // -> [NR_PASSES]
  actor->push_back(torch::nn::Softmax(/*dim=*/0u));       // -> [NR_PASSES]
  return actor;
}
torch::nn::Sequential
make_LSTM_critic(unsigned int max_qubits, unsigned int hidden_size_multiplier,
                 unsigned int projection_size_multiplier) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  const unsigned int H = hidden_size_multiplier * IRS;
  const unsigned int P = projection_size_multiplier * IRS;
  auto critic = torch::nn::Sequential();
  if (H == P) {
    critic->push_back(torch::nn::LSTM(
        torch::nn::LSTMOptions(/*input_size=*/IRS, /*hidden_size=*/H)
            .bidirectional(true))); // -> [N, (2*P, (h, c))]
  } else {
    critic->push_back(torch::nn::LSTM(
        torch::nn::LSTMOptions(/*input_size=*/IRS, /*hidden_size=*/H)
            .bidirectional(true)
            .proj_size(P))); // -> [N, (2*P, (h, c))]
  }
  critic->push_back(torch::nn::FilterLSTM());                // -> [N, 2*P]
  critic->push_back(torch::nn::TransposeContiguous(0u, 1u)); // -> [2*P, N]
  critic->push_back(torch::nn::AdaptiveAvgPool1d(1u));       // -> [2*P, 1]
  critic->push_back(torch::nn::Flatten(
      torch::nn::FlattenOptions().start_dim(/*dim=*/0))); // -> [2*P]
  critic->push_back(torch::nn::Linear(2 * P, 1));         // -> [1]
  critic->push_back(torch::nn::Squeeze(/*dim=*/0));       // -> []
  return critic;
}

torch::nn::Sequential
make_hybrid_actor(unsigned int max_qubits, unsigned int nr_residual_blocks,
                  unsigned int kernel_size, unsigned int hidden_size_multiplier,
                  unsigned int projection_size_multiplier,
                  const std::optional<double> &optional_prelu_init) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  const unsigned int H = hidden_size_multiplier * IRS;
  const unsigned int P = projection_size_multiplier * IRS;
  auto actor = torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u)); // -> [IRS, N]
  if (optional_prelu_init.has_value()) {
    actor->push_back(
        TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                optional_prelu_init.value())); // -> [IRS, N]
  } else {
    actor->push_back(TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                            kernel_size)); // -> [IRS, N]
  }

  actor->push_back(torch::nn::TransposeContiguous(0u, 1u)); // -> [N, IRS]
  if (H == P) {
    actor->push_back(torch::nn::LSTM(
        torch::nn::LSTMOptions(/*input_size=*/IRS, /*hidden_size=*/H)
            .bidirectional(true))); // -> [N, 2*P]
  } else {
    actor->push_back(torch::nn::LSTM(
        torch::nn::LSTMOptions(/*input_size=*/IRS, /*hidden_size=*/H)
            .bidirectional(true)
            .proj_size(P))); // -> [N, 2*P]
  }
  actor->push_back(torch::nn::FilterLSTM());
  actor->push_back(torch::nn::TransposeContiguous(0u, 1u)); // -> [2*P, N]
  actor->push_back(torch::nn::AdaptiveAvgPool1d(1u));       // -> [2*P, 1]
  actor->push_back(torch::nn::Flatten(
      torch::nn::FlattenOptions().start_dim(/*dim=*/0))); // -> [2*P]
  actor->push_back(torch::nn::Linear(2 * P, NR_PASSES));  // -> [NR_PASSES]
  actor->push_back(torch::nn::Softmax(/*dim=*/0u));       // -> [NR_PASSES]
  return actor;
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
  auto critic = torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u)); // -> [IRS, N]
  if (optional_prelu_init.has_value()) {
    critic->push_back(
        TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                                optional_prelu_init.value())); // -> [IRS, N]
  } else {
    critic->push_back(TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                                             kernel_size)); // -> [IRS, N]
  }
  critic->push_back(torch::nn::TransposeContiguous(0u, 1u)); // -> [N, IRS]
  if (H == P) {
    critic->push_back(torch::nn::LSTM(
        torch::nn::LSTMOptions(/*input_size=*/IRS, /*hidden_size=*/H)
            .bidirectional(true))); // -> [N, (2*P, (h, c))]
  } else {
    critic->push_back(torch::nn::LSTM(
        torch::nn::LSTMOptions(/*input_size=*/IRS, /*hidden_size=*/H)
            .bidirectional(true)
            .proj_size(P))); // -> [N, (2*P, (h, c))]
  }
  critic->push_back(torch::nn::FilterLSTM());                // -> [N, 2*P]
  critic->push_back(torch::nn::TransposeContiguous(0u, 1u)); // -> [2*P, N]
  critic->push_back(torch::nn::AdaptiveAvgPool1d(1u));       // -> [2*P, 1]
  critic->push_back(torch::nn::Flatten(
      torch::nn::FlattenOptions().start_dim(/*dim=*/0))); // -> [2*P]
  critic->push_back(torch::nn::Linear(2 * P, 1));         // -> [1]
  critic->push_back(torch::nn::Squeeze(/*dim=*/0));       // -> []
  return critic;
}

} // namespace ai_pass_selector
