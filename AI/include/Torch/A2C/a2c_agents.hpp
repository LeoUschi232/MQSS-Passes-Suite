#ifndef A2C_AGENTS_HPP
#define A2C_AGENTS_HPP

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// Torch includes
#include "Torch/A2C/base_a2c_agent.hpp"

// Utils includes
#include "Utils/circuit_utils.hpp"

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
// Low-dimensional instruction representation sizes are arbitrarily chosen.
constexpr unsigned int TINY_LOWDIM_INSTR_REPR_SIZE = 8;
constexpr unsigned int SMALL_LOWDIM_INSTR_REPR_SIZE = 10;
constexpr unsigned int MODERATE_LOWDIM_INSTR_REPR_SIZE = 15;
constexpr unsigned int BIG_LOWDIM_INSTR_REPR_SIZE = 21;
constexpr unsigned int HUGE_LOWDIM_INSTR_REPR_SIZE = 22;

const std::unordered_map<int, unsigned int>
    CIRCUIT_SIZE_CLASS_TO_LOWDIM_INSTR_REPR_SIZE = {
        {TINY, TINY_LOWDIM_INSTR_REPR_SIZE},
        {SMALL, SMALL_LOWDIM_INSTR_REPR_SIZE},
        {MODERATE, MODERATE_LOWDIM_INSTR_REPR_SIZE},
        {BIG, BIG_LOWDIM_INSTR_REPR_SIZE},
        {HUGE, HUGE_LOWDIM_INSTR_REPR_SIZE}};

/// A2C = Advantage Actor-Critic
/// IB = Instruction-Based
/// DB = Depth-Based
/// CONV{X} = Convolutional with kernel size X×Instruction-representation-size
DECLARE_A2C_AGENT(A2C_IBCONV2)
DECLARE_A2C_AGENT(A2C_IBCONV4)
} // namespace ai_pass_selector
#endif // A2C_TRAINER_HPP
