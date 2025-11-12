#ifndef BASE_PPO_AGENT_HPP
#define BASE_PPO_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Neural-Networks includes
#include "NeuralNetworks/Agents/base_actor_critic.hpp"

namespace fs = std::filesystem;

namespace ai_pass_selector {
enum class OptimizerType : int;

class BasePPOAgent : public BaseActorCritic {
protected:
  /// PPO specific attributes
  double min_ratio = 0.0;
  double max_ratio = 0.0;

public:
  /// Constructors
  explicit BasePPOAgent(unsigned int max_qubits);

  /// Destructor
  ~BasePPOAgent() override = default;

  /// Copy and move constructors and assignment operators
  BasePPOAgent(const BasePPOAgent &other) noexcept = delete;

  BasePPOAgent(BasePPOAgent &&other) noexcept = default;

  BasePPOAgent &operator=(const BasePPOAgent &other) noexcept = delete;

  BasePPOAgent &operator=(BasePPOAgent &&other) noexcept = default;

  /**
   * @param actor
   * @param critic
   * @return
   */
  bool initialize(const torch::nn::Sequential &actor,
                  const torch::nn::Sequential &critic);

  //////////////////////////////////////////////////////////////////////////////
  /// PPO standard methods
  /**
   * @param observation
   * @return
   */
  std::pair<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &observation);

  /**
   * @param observation
   * @param action_index_unsqueezed
   * @return [new_log_action_prob, new_state_value, entropy]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  force_select_action(const torch::Tensor &observation,
                      const torch::Tensor &action_index_unsqueezed);

  /**
   * @param observation
   * @return
   */
  torch::Tensor get_value(const torch::Tensor &observation);

  /**
   * @param observation
   * @return [action_index, log_action_probs, state_values, entropy]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  select_action(const torch::Tensor &observation);

  /**
   *
   * @param old_log_action_probs
   * @param old_state_values
   * @param new_log_action_probs
   * @param new_state_values
   * @param rewards
   * @param entropy
   * @return [actor_loss, critic_loss]
   */
  std::pair<torch::Tensor, torch::Tensor>
  get_losses(const torch::Tensor &old_log_action_probs,
             const torch::Tensor &old_state_values,
             const torch::Tensor &new_log_action_probs,
             const torch::Tensor &new_state_values,
             const torch::Tensor &rewards, const torch::Tensor &entropy);

  /**
   * @param actor_loss
   * @param critic_loss
   */
  void update_parameters(const torch::Tensor &actor_loss,
                         const torch::Tensor &critic_loss) const;
  //////////////////////////////////////////////////////////////////////////////
};
} // namespace ai_pass_selector

#endif // BASE_PPO_AGENT_HPP