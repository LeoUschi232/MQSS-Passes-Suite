#ifndef BASE_SDSAC_AGENT_HPP
#define BASE_SDSAC_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Neural-Networks includes
#include "NeuralNetworks/Agents/base_actor_critic.hpp"

// Standard library includes
#include <memory>
namespace fs = std::filesystem;

namespace ai_pass_selector {
enum class OptimizerType : int;

class BaseSDSACAgent : public BaseActorCritic {
protected:
  /// SDSAC specific attributes
  torch::Tensor sdsac_temperature_alpha = torch::tensor(0.0);
  double sdsac_shared_learning_rate = 0.0;
  double sdsac_smoothing_tau = 0.0;
  double sdsac_penalty_beta = 0.0;
  double sdsac_clip_c = 0.0;

  /// SDSAC Additional Critics
  torch::nn::Sequential critic_Q2_main{nullptr};
  torch::nn::Sequential critic_Q1_avg{nullptr};
  torch::nn::Sequential critic_Q2_avg{nullptr};
  std::shared_ptr<torch::optim::Optimizer> critic_Q1_optimizer = nullptr;
  std::shared_ptr<torch::optim::Optimizer> critic_Q2_optimizer = nullptr;
  std::shared_ptr<torch::optim::Adam> alpha_optimizer = nullptr;

public:
  /// Constructors
  explicit BaseSDSACAgent(unsigned int max_qubits);

  /// Destructor
  ~BaseSDSACAgent() override = default;

  /// Copy and move constructors and assignment operators
  BaseSDSACAgent(const BaseSDSACAgent &other) noexcept = delete;

  BaseSDSACAgent(BaseSDSACAgent &&other) noexcept = default;

  BaseSDSACAgent &operator=(const BaseSDSACAgent &other) noexcept = delete;

  BaseSDSACAgent &operator=(BaseSDSACAgent &&other) noexcept = delete;

  //////////////////////////////////////////////////////////////////////////////
  /// SDSAC override for disabling methods
  bool initialize(const torch::nn::Sequential &actor,
                  const torch::nn::Sequential &critic) override;
  void update_parameters(const torch::Tensor &actor_loss,
                         const torch::Tensor &critic_loss) const override;
  std::pair<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &observation) override;
  torch::Tensor get_value(const torch::Tensor &observation) override;
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  select_action(const torch::Tensor &observation) override;
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  /// SDSAC quadruple-critic methods
  /**
   *
   * @param actor
   * @param critic_Q1_main
   * @param critic_Q2_main
   * @param critic_Q1_avg
   * @param critic_Q2_avg
   * @return
   */
  bool initialize(const torch::nn::Sequential &actor,
                  const torch::nn::Sequential &critic_Q1_main,
                  const torch::nn::Sequential &critic_Q2_main,
                  const torch::nn::Sequential &critic_Q1_avg,
                  const torch::nn::Sequential &critic_Q2_avg);

  /**
   *
   * @param observation
   * @return [policy, Q1_main, Q2_main, Q1_avg, Q2_avg]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
  sdsac_forward(const torch::Tensor &observation);

  /**
   *
   * @param observation
   * @return [action_index, entropy]
   */
  std::pair<torch::Tensor, torch::Tensor>
  sdsac_select_action(const torch::Tensor &observation);

  /**
   *
   * @param new_action_probs
   * @param old_entropy
   * @param new_entropy
   * @param reward
   * @param old_action
   * @param Q1_main
   * @param Q2_main
   * @param Q1_avg
   * @param Q2_avg
   * @param next_action_probs
   * @param next_Q1_avg
   * @param next_Q2_avg
   * @param done
   * @return [actor_loss, critic_Q1_loss, critic_Q2_loss,
   * optional_temperature_alpha_loss]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  get_loss(const torch::Tensor &new_action_probs,
           const torch::Tensor &old_entropy, const torch::Tensor &new_entropy,
           const torch::Tensor &reward, const torch::Tensor &old_action,
           const torch::Tensor &Q1_main, const torch::Tensor &Q2_main,
           const torch::Tensor &Q1_avg, const torch::Tensor &Q2_avg,
           const torch::Tensor &next_action_probs,
           const torch::Tensor &next_Q1_avg, const torch::Tensor &next_Q2_avg,
           const torch::Tensor &done);

  /**
   *
   * @param actor_loss
   * @param critic_Q1_loss
   * @param critic_Q2_loss
   * @param temperature_alpha_loss
   */
  void
  sdsac_update_parameters(const torch::Tensor &actor_loss,
                          const torch::Tensor &critic_Q1_loss,
                          const torch::Tensor &critic_Q2_loss,
                          const torch::Tensor &temperature_alpha_loss) const;
  //////////////////////////////////////////////////////////////////////////////
};

} // namespace ai_pass_selector

#endif // BASE_SDSAC_AGENT_HPP