#ifndef BASE_A2C_AGENT_HPP
#define BASE_A2C_AGENT_HPP

// Torch includes
#include <torch/torch.h>

// Standard library includes
#include <utility>
#include <tuple>
#include <memory>


namespace ai_pass_selector {
class BaseA2CAgent : public torch::nn::Module {
  const int max_qubits;
  const int max_instructions;
  const int max_depth;
  const int nr_input_values;
  const int nr_output_values;
  const double critic_learning_rate;
  const double actor_learning_rate;
  const int nr_parallel_environments;
  torch::nn::Sequential critic;
  torch::nn::Sequential actor;
  std::unique_ptr<torch::optim::Optimizer> actor_optimizer;
  std::unique_ptr<torch::optim::Optimizer> critic_optimizer;
  torch::Device device;

public:
  /// Constructor
  BaseA2CAgent(
      int max_qubits,
      int max_instructions,
      int max_depth,
      int nr_input_values,
      int nr_output_values,
      const torch::nn::Sequential &critic,
      const torch::nn::Sequential &actor,
      int critic_optimizer_type,
      int actor_optimizer_type,
      double critic_learning_rate = 0.005,
      double actor_learning_rate = 0.001,
      int nr_parallel_environments = 10,
      torch::Device device = torch::kCPU);

  /// Destructor
  ~BaseA2CAgent() override = default;

  /// Copy and move constructors and assignment operators
  BaseA2CAgent(const BaseA2CAgent &other) = delete;

  BaseA2CAgent(BaseA2CAgent &&other) noexcept = default;

  BaseA2CAgent &operator=(const BaseA2CAgent &other) = delete;

  BaseA2CAgent &operator=(BaseA2CAgent &&other) noexcept = delete;

  /**
   *
   * @param batched_observations
   * @return
   */
  std::pair<torch::Tensor, torch::Tensor> forward(
      torch::Tensor batched_observations);

  /**
   *
   * @param batched_observations
   * @return
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  select_action(const torch::Tensor &batched_observations);

  /**
   *
   * @param rewards
   * @param log_action_probs
   * @param state_values
   * @param entropy
   * @param termination_masks
   * @param discount_factor
   * @param gae_hyperparameter
   * @param entropy_coefficient
   * @return
   */
  static std::pair<torch::Tensor, torch::Tensor> get_losses(
      const torch::Tensor &rewards,
      const torch::Tensor &log_action_probs,
      const torch::Tensor &state_values,
      const torch::Tensor &entropy,
      const torch::Tensor &termination_masks,
      double discount_factor,
      double gae_hyperparameter,
      double entropy_coefficient);


  /**
   *
   * @param critic_loss
   * @param actor_loss
   */
  void update_parameters(const torch::Tensor &critic_loss,
                         const torch::Tensor &actor_loss) const;

  /**
   *
   */
  virtual void save_model() const = 0;

  /**
   *
   */
  virtual void load_model() = 0;

};

} // namespace ai_pass_selector

#endif // BASE_A2C_AGENT_HPP