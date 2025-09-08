#ifndef A2C_AGENT_HPP
#define A2C_AGENT_HPP

// Torch includes
#include <torch/torch.h>

// Standard library includes
#include <utility>
#include <tuple>
#include <memory>


namespace ai_pass_selector {
class A2CAgent final : torch::nn::Module {
  const int nr_input_values;
  const int nr_actions;
  const int max_qubits;
  const int max_instructions;
  const int max_depth;
  const double critic_learning_rate;
  const double actor_learning_rate;
  const int nr_parallel_environments;
  torch::nn::Sequential critic;
  torch::nn::Sequential actor;
  std::unique_ptr<torch::optim::Adam> actor_optimizer;
  std::unique_ptr<torch::optim::Adam> critic_optimizer;
  torch::Device device;

public:
  /// Constructor
  A2CAgent(
      int nr_input_values,
      int nr_actions,
      int max_qubits,
      int max_instructions,
      int max_depth,
      double critic_learning_rate = 0.005,
      double actor_learning_rate = 0.001,
      int nr_parallel_environments = 10,
      torch::Device device = torch::kCPU
      );

  /// Destructor
  ~A2CAgent() override = default;

  /// Copy and move constructors and assignment operators
  A2CAgent(const A2CAgent &other) = delete;

  A2CAgent(A2CAgent &&other) noexcept = default;

  A2CAgent &operator=(const A2CAgent &other) = delete;

  A2CAgent &operator=(A2CAgent &&other) noexcept = delete;

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
   * @param dir
   * @param who
   * @return
   */
  std::string make_path(const std::string &dir, const std::string &who) const;

  /**
   *
   * @param critic_loss
   * @param actor_loss
   */
  void update_parameters(const torch::Tensor &critic_loss,
                         const torch::Tensor &actor_loss);

  /**
   *
   * @param weights_dir
   */
  void save_model(const std::string &weights_dir) const;

  /**
   *
   * @param weights_dir
   */
  void load_model(const std::string &weights_dir);

};

} // namespace ai_pass_selector

#endif //A2C_AGENT_HPP