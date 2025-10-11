#ifndef A3C_AGENTS_HPP
#define A3C_AGENTS_HPP

// Torch includes
#include "Agents/A3C/base_a3c_agent.hpp"

#define DECLARE_A3C_AGENT(ClassName)                                           \
  class ClassName final : public ai_pass_selector::BaseA3CAgent {              \
  public:                                                                      \
    ClassName(unsigned int max_qubits,                                         \
              std::unordered_map<std::string, std::string> params,             \
              bool is_boss = true);                                            \
    std::string agentName() const override;                                    \
    std::unique_ptr<BaseA3CAgent> clone() const override;                      \
  };

namespace torch::nn {
/// Custom torch LeakyReLU layer with learnable parameter for negative inputs.
inline PReLU HalfScalingLayer(double prelu_init = 1.0) {
  return PReLU(PReLUOptions().init(prelu_init));
}
/// Instruction tensor will have shape {B, N, IRS}
/// B = Batch size / Nr of parallel environments
/// N = Nr of instructions in the quantum circuit
/// IRS = Instruction Representation Size
inline Functional TransposeContiguous(int64_t dim0, int64_t dim1) {
  return Functional([dim0, dim1](const Tensor &x) {
    return x.transpose(dim0, dim1).contiguous();
  });
}
inline Functional Transpose(int64_t dim0, int64_t dim1) {
  return Functional(
      [dim0, dim1](const Tensor &x) { return x.transpose(dim0, dim1); });
}
} // namespace torch::nn

namespace ai_pass_selector {
/// A3C = Asynchronous Advantage Actor-Critic
/// TCN = Temporal Convolutional Network
DECLARE_A3C_AGENT(A3C_TCN)
} // namespace ai_pass_selector
#endif // A3C_TRAINER_HPP
