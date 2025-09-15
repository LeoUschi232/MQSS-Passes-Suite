#ifndef INSTRUCTION_BASED_A2C_HPP
#define INSTRUCTION_BASED_A2C_HPP
#include "Torch/A2C/base_a2c_agent.hpp"

#include "Torch/agent_utils..hpp"

namespace ai_pass_selector {
/// IB = Instruction Based
/// DB = Instruction Based
/// FC = Fully Connected
/// CONV = Convolutional
/// LSM = Layer Size Maintaining
/// LSD = Layer Size Decreasing
class IB_FC_LSD_A2C final : public BaseA2CAgent {
public:
  IB_FC_LSD_A2C(
      unsigned int max_qubits,
      unsigned int max_instructions,
      unsigned int max_depth,
      int critic_optimizer_type = OPTIMIZER_ADAM,
      int actor_optimizer_type = OPTIMIZER_ADAM,
      double critic_learning_rate = 0.005,
      double actor_learning_rate = 0.001,
      unsigned int nr_parallel_environments = 10,
      torch::Device device = torch::kCPU);

  std::string model_name() const override;
};

class IB_FC_LSM_A2C final : public BaseA2CAgent {
public:
  IB_FC_LSM_A2C(
      unsigned int max_qubits,
      unsigned int max_instructions,
      unsigned int max_depth,
      int critic_optimizer_type = OPTIMIZER_ADAM,
      int actor_optimizer_type = OPTIMIZER_ADAM,
      double critic_learning_rate = 0.02,
      double actor_learning_rate = 0.004,
      unsigned int nr_parallel_environments = 5,
      torch::Device device = torch::kCPU);

  std::string model_name() const override;
};

} // namespace ai_pass_selector

#endif // INSTRUCTION_BASED_A2C_HPP