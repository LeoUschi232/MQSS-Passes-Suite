#ifndef A2C_AGENTS_HPP
#define A2C_AGENTS_HPP

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// Torch includes
#include "Torch/A2C/base_a2c_agent.hpp"

#define DECLARE_A2C_AGENT(ClassName)                                           \
  class ClassName final : public ai_pass_selector::BaseA2CAgent {              \
  public:                                                                      \
    ClassName(unsigned int max_qubits,                                         \
              std::unordered_map<std::string, std::string> params);            \
    std::string agentName() const override;                                    \
  };

namespace torch::nn {
/// Instruction tensor will have shape {B, N, IRP}
/// B = Batch size / Nr of parallel environments
/// N = Nr of instructions in the quantum circuit
/// IRP = Instruction representation size
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
/// A2C = Advantage Actor-Critic
/// CONV{X} = Convolutional with depth X
/// NPI = Normlize per instruction
/// NFULL = Normlize the full chain of instructions
DECLARE_A2C_AGENT(A2C_CONV2_NPI)
DECLARE_A2C_AGENT(A2C_CONV2_NFULL)
} // namespace ai_pass_selector
#endif // A2C_TRAINER_HPP
