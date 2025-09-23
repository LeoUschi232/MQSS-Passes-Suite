#ifndef A2C_AGENTS_HPP
#define A2C_AGENTS_HPP

#include "Torch/A2C/base_a2c_agent.hpp"

#define DECLARE_A2C_AGENT(ClassName)                                           \
  class ClassName final : public ai_pass_selector::BaseA2CAgent {              \
  public:                                                                      \
    ClassName(int circuit_size_class,                                          \
              std::unordered_map<std::string, std::string> params);            \
    ClassName(const std::string &circuit_size,                                 \
              std::unordered_map<std::string, std::string> params);            \
    std::string agentName() const override;                                    \
  };

namespace torch::nn {
/// Custom torch LeakyReLU layer with learnable parameter for negative inputs.
inline PReLU HalfScalingLayer(int num_parameters, double init = 1.0) {
  return PReLU(PReLUOptions().num_parameters(num_parameters).init(init));
}
/// Custom torch Conv1d layer with automatic output size calculation.
inline Conv1d ConvolutionalLayer(unsigned int L_in, unsigned int L_out,
                                 unsigned int kernel_size) {
  // This layer assumes each instruction with its corresponding attributes:
  // qubit triggers, gate triggers and params maps exclusively to its own L_out
  // number of kernels.
  return Conv1d(Conv1dOptions(L_in, L_out, kernel_size).stride(kernel_size));
}
} // namespace torch::nn

namespace ai_pass_selector {
/// A2C = Advantage Actor-Critic
/// IB = Instruction-Based
/// DB = Depth-Based
/// FC = Fully Connected
/// CONV = Convolutional
/// LSM = Layer Size Maintaining
/// LSD = Layer Size Decreasing
DECLARE_A2C_AGENT(A2C_IB_FC_LSM)
DECLARE_A2C_AGENT(A2C_IB_FC_LSD)
DECLARE_A2C_AGENT(A2C_IB_CONV_LSM)
DECLARE_A2C_AGENT(A2C_IB_CONV_LSD)
DECLARE_A2C_AGENT(A2C_DB_FC_LSM)
DECLARE_A2C_AGENT(A2C_DB_FC_LSD)
DECLARE_A2C_AGENT(A2C_DB_CONV_LSM)
DECLARE_A2C_AGENT(A2C_DB_CONV_LSD)
} // namespace ai_pass_selector
#endif // A2C_TRAINER_HPP
