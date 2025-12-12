#ifndef BASE_A2C_AGENT_HPP
#define BASE_A2C_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Neural-Networks includes
#include "NeuralNetworks/Agents/base_actor_critic.hpp"

namespace fs = std::filesystem;

namespace ai_pass_selector {
enum class OptimizerType : int;

class BaseA2CAgent : public BaseActorCritic {
public:
  /// Constructors
  explicit BaseA2CAgent(unsigned int max_qubits);

  /// Destructor
  ~BaseA2CAgent() override = default;

  /// Copy and move constructors and assignment operators
  BaseA2CAgent(const BaseA2CAgent &other) noexcept = delete;

  BaseA2CAgent(BaseA2CAgent &&other) noexcept = default;

  BaseA2CAgent &operator=(const BaseA2CAgent &other) noexcept = delete;

  BaseA2CAgent &operator=(BaseA2CAgent &&other) noexcept = delete;

  /**
   * @param actor
   * @param critic
   * @return
   */
  bool initialize(const torch::nn::Sequential &actor,
                  const torch::nn::Sequential &critic);

  //////////////////////////////////////////////////////////////////////////////
  /// A2C/A2C standard methods
  /**
   * @param observation
   * @return
   */
  std::pair<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &observation);

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
   * @param log_action_probs
   * @param state_values
   * @param final_state_value
   * @param rewards
   * @param entropy
   * @return [actor_loss, critic_loss]
   */
  std::pair<torch::Tensor, torch::Tensor>
  get_losses(const torch::Tensor &log_action_probs, // Shape [T]
             const torch::Tensor &state_values, // Shape [T]
             const torch::Tensor &final_state_value, // Shape []
             const torch::Tensor &rewards, // Shape [T]
             const torch::Tensor &entropy // Shape [T]
    );

  /**
   * @param actor_loss
   * @param critic_loss
   */
  void update_parameters(const torch::Tensor &actor_loss,
                         const torch::Tensor &critic_loss) const;
  //////////////////////////////////////////////////////////////////////////////
};
} // namespace ai_pass_selector

#endif // BASE_A2C_AGENT_HPP
