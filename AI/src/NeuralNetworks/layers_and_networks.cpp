#include "NeuralNetworks/layers_and_networks.hpp"

// Torch includes
#include "torch/torch.h"
namespace ai_pass_selector {
torch::nn::Sequential make_TCN_actor(unsigned int max_qubits){

}
torch::nn::Sequential make_TCN_critic(unsigned int max_qubits);
torch::nn::Sequential make_LSTM_actor(unsigned int max_qubits);
torch::nn::Sequential make_LSTM_critic(unsigned int max_qubits);
torch::nn::Sequential make_HYBRID_actor(unsigned int max_qubits);
torch::nn::Sequential make_HYBRID_critic(unsigned int max_qubits);
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
