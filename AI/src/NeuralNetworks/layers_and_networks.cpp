#include "NeuralNetworks/layers_and_networks.hpp"

// Neural-Networks includes
#include "NeuralNetworks/tcn_full_network.hpp"

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/passes_utils.hpp"

namespace ai_pass_selector {
torch::nn::Sequential
make_TCN_actor(unsigned int max_qubits,
               const std::optional<double> &optional_prelu_init) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  constexpr unsigned int nr_residual_blocks = 12u;
  constexpr unsigned int kernel_size = 5u;
  if (!optional_prelu_init.has_value()) {
    return torch::nn::Sequential(
        torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
        TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                               kernel_size), // -> [IRS, N]
        torch::nn::AdaptiveAvgPool1d(1u),    // -> [IRS, 1]
        torch::nn::Flatten(
            torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [NR_PASSES]
        torch::nn::Linear(IRS, NR_PASSES),                     // -> [NR_PASSES]
        torch::nn::ReLU(),                                     // -> [NR_PASSES]
        torch::nn::Softmax(/*dim=*/0u)                         // -> [NR_PASSES]
    );
  }
  double prelu_init = optional_prelu_init.value();
  return torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                              prelu_init), // -> [IRS, N]
      torch::nn::AdaptiveAvgPool1d(1u),    // -> [IRS, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [NR_PASSES]
      torch::nn::Linear(IRS, NR_PASSES),                     // -> [NR_PASSES]
      torch::nn::PReLU(
          torch::nn::PReLUOptions().init(prelu_init)), // -> [NR_PASSES]
      torch::nn::Softmax(/*dim=*/0u)                   // -> [NR_PASSES]
  );
}
torch::nn::Sequential
make_TCN_critic(unsigned int max_qubits,
                const std::optional<double> &optional_prelu_init) {
  const unsigned int IRS = MAX_QUBITS_TO_IRS(max_qubits);
  constexpr unsigned int nr_residual_blocks = 12u;
  constexpr unsigned int kernel_size = 5u;
  if (!optional_prelu_init.has_value()) {
    return torch::nn::Sequential(
        torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
        TCNFullNetworkWithReLU(IRS, nr_residual_blocks,
                               kernel_size), // -> [IRS, N]
        torch::nn::AdaptiveAvgPool1d(1u),    // -> [IRS, 1]
        torch::nn::Flatten(
            torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [NR_PASSES]
        torch::nn::Linear(IRS, 1),                             // -> [1]
        torch::nn::ReLU()                                      // -> [1]
    );
  }
  double prelu_init = optional_prelu_init.value();
  return torch::nn::Sequential(
      torch::nn::TransposeContiguous(0u, 1u), // -> [IRS, N]
      TCNFullNetworkWithPReLU(IRS, nr_residual_blocks, kernel_size,
                              prelu_init), // -> [IRS, N]
      torch::nn::AdaptiveAvgPool1d(1u),    // -> [IRS, 1]
      torch::nn::Flatten(
          torch::nn::FlattenOptions().start_dim(/*dim=*/0)), // -> [NR_PASSES]
      torch::nn::Linear(IRS, 1),                             // -> [1]
      torch::nn::PReLU(torch::nn::PReLUOptions().init(prelu_init)) // -> [1]
  );
}
torch::nn::Sequential
make_LSTM_actor(unsigned int max_qubits,
                const std::optional<double> &optional_prelu_init) {
  throw std::runtime_error("Not implemented yet.");
}
torch::nn::Sequential
make_LSTM_critic(unsigned int max_qubits,
                 const std::optional<double> &optional_prelu_init) {
  throw std::runtime_error("Not implemented yet.");
}
torch::nn::Sequential
make_HYBRID_actor(unsigned int max_qubits,
                  const std::optional<double> &optional_prelu_init) {
  throw std::runtime_error("Not implemented yet.");
}
torch::nn::Sequential
make_HYBRID_critic(unsigned int max_qubits,
                   const std::optional<double> &optional_prelu_init) {
  throw std::runtime_error("Not implemented yet.");
}
} // namespace ai_pass_selector

namespace torch::nn {
Functional TransposeContiguous(int32_t dim0, int32_t dim1) {
  return Functional([dim0, dim1](const Tensor &x) {
    return x.transpose(dim0, dim1).contiguous();
  });
}

Functional Transpose(int32_t dim0, int32_t dim1) {
  return Functional(
      [dim0, dim1](const Tensor &x) { return x.transpose(dim0, dim1); });
}

Functional FiniteCheck(std::string stage_name) {
  return Functional([name = std::move(stage_name)](const Tensor &tensor) {
    if (const bool has_inf = tensor.isinf().any().item<bool>(),
        has_nan = tensor.isnan().any().item<bool>();
        has_nan || has_inf) {
      const double min_val = tensor.amin().item<double>();
      const double max_val = tensor.amax().item<double>();
      std::cerr << "[FiniteCheck] " << name << " nan=" << has_nan
                << " inf=" << has_inf << " min=" << min_val
                << " max=" << max_val << "\n";
    }
    return tensor;
  });
}

Functional ShapeProbe(std::string stage_name) {
  return Functional([name = std::move(stage_name)](const Tensor &x) {
    std::ostringstream oss;
    oss << "[ShapeProbe] " << name << " sizes=[";
    for (size_t i = 0; i < x.sizes().size(); ++i) {
      oss << x.sizes()[i] << (i + 1 < x.sizes().size() ? "," : "");
    }
    oss << "] min=" << x.amin().item<double>()
        << " max=" << x.amax().item<double>()
        << " finite=" << x.isfinite().all().item<bool>() << "\n";
    std::cerr << oss.str();
    return x;
  });
}

} // namespace torch::nn
