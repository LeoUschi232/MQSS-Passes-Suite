#ifndef BASE_ACER_AGENT_HPP
#define BASE_ACER_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Neural-Networks includes
#include "NeuralNetworks/Agents/base_actor_critic.hpp"

namespace fs = std::filesystem;

namespace ai_pass_selector {
constexpr float DIVISION_BY_ZERO_BLOCK = 1e-12f;
enum class OptimizerType : int;

class BaseACERAgent : public BaseActorCritic {
protected:
  /// ACER specific attributes
  torch::Tensor acer_truncation_threshold_c = torch::zeros({});
  torch::Tensor acer_trust_region_delta = torch::zeros({});
  double acer_soft_update_alpha = 0.0;

  /// ACER Additional Actor
  torch::nn::Sequential actor_avg{nullptr};

public:
  /// Constructors
  explicit BaseACERAgent(unsigned int max_qubits);

  /// Destructor
  ~BaseACERAgent() override = default;

  /// Copy and move constructors and assignment operators
  BaseACERAgent(const BaseACERAgent &other) noexcept = delete;

  BaseACERAgent(BaseACERAgent &&other) noexcept = default;

  BaseACERAgent &operator=(const BaseACERAgent &other) noexcept = delete;

  BaseACERAgent &operator=(BaseACERAgent &&other) noexcept = delete;

  /**
   * @param actor_main
   * @param actor_avg
   * @param critic_Q_estimator
   * @return
   */
  bool initialize(const torch::nn::Sequential &actor_main,
                  const torch::nn::Sequential &actor_avg,
                  const torch::nn::Sequential &critic_Q_estimator);

  //////////////////////////////////////////////////////////////////////////////
  /// ACER standard methods
  void reset_gradients();

  /**
   * @param observation
   * @return [policy_main, policy_avg, Q_values]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &observation);

  /**
   * @param observation
   * @return
   */
  torch::Tensor get_value_main(const torch::Tensor &observation);

  /**
   * @param observation
   * @return
   */
  torch::Tensor get_value_avg(const torch::Tensor &observation);

  /**
   * @param action_probs
   * @return [action, entropy]
   */
  torch::Tensor select_action(const torch::Tensor &action_probs);

  /**
   * @param observation
   * @return
   */
  unsigned int select_greedy_action(const torch::Tensor &observation) override;

  /**
   * @param reward
   * @param Q_ret
   * @param policy_main
   * @param policy_avg
   * @param Q_values
   * @param original_policy
   * @param action_index
   * @return [actor_gradients, critic_loss], new_Q_ret
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  compute_losses_and_accumulate_gradients(
      const torch::Tensor &reward,          // Shape []
      torch::Tensor Q_ret,                  // Shape []
      const torch::Tensor &policy_main,     // Shape [NR_PASSES]
      const torch::Tensor &policy_avg,      // Shape [NR_PASSES]
      const torch::Tensor &Q_values,        // Shape [NR_PASSES]
      const torch::Tensor &original_policy, // Shape [NR_PASSES]
      unsigned int action_index             // Shape []
  );

  /**
   *
   * @param actor_gradients
   * @param critic_loss
   */
  void update_parameters(const torch::Tensor &actor_gradients,
                         const torch::Tensor &critic_loss);
  //////////////////////////////////////////////////////////////////////////////
  /// Saving and Loading
  void save_model() const override;
  void load_model() override;
};
} // namespace ai_pass_selector

#endif // BASE_ACER_AGENT_HPP
