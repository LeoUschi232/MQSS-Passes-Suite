#ifndef BASE_ACER_AGENT_HPP
#define BASE_ACER_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Neural-Networks includes
#include "NeuralNetworks/Agents/base_actor_critic.hpp"

namespace fs = std::filesystem;

namespace ai_pass_selector {
enum class OptimizerType : int;

class BaseACERAgent : public BaseActorCritic {
protected:
  /// ACER specific attributes
  double acer_truncation_threshold_c = 0.0;
  double acer_trust_region_delta = 0.0;

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
  torch::Tensor get_value(const torch::Tensor &observation);

  /**
   * @param action_probs
   * @return [action, entropy]
   */
  std::pair<torch::Tensor, torch::Tensor>
  select_action(const torch::Tensor &action_probs);

  /**
   * @param observation
   * @return
   */
  unsigned int select_greedy_action(const torch::Tensor &observation) override;

  //////////////////////////////////////////////////////////////////////////////
  /// Saving and Loading
  void save_model() const override;
};
} // namespace ai_pass_selector

#endif // BASE_ACER_AGENT_HPP
