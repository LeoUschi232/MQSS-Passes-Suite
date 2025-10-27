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
  bool ppo_critic_loss_on_advantages = false;

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

  //////////////////////////////////////////////////////////////////////////////
  /// PPO standard methods

  /**
   * @param observation
   * @param action_index_unsqueezed
   * @return [new_log_action_prob, new_state_value, entropy]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  force_select_action(const torch::Tensor &observation,
                      const torch::Tensor &action_index_unsqueezed);

  /**
   *
   * @param old_advantages
   * @param old_log_action_probs
   * @param new_log_action_probs
   * @param new_state_values
   * @param rewards
   * @param entropy
   * @return [actor_loss, critic_loss]
   */
  std::pair<torch::Tensor, torch::Tensor>
  get_losses(const torch::Tensor &old_advantages,
             const torch::Tensor &old_log_action_probs,
             const torch::Tensor &new_log_action_probs,
             const torch::Tensor &new_state_values,
             const torch::Tensor &rewards,
             const torch::Tensor &entropy);

  /**
   *
   * @param total_loss
   */
  void update_parameters(const torch::Tensor &total_loss) const;
  //////////////////////////////////////////////////////////////////////////////

  /// Saving and Loading
  void save_model() const override;
};
} // namespace ai_pass_selector

#endif // BASE_PPO_AGENT_HPP