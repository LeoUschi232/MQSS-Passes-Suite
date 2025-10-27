#ifndef BASE_ACTOR_CRITIC_HPP
#define BASE_ACTOR_CRITIC_HPP

// Agents includes
#include "NeuralNetworks/Agents/abstract_agent.hpp"

// Torch includes
#include "torch/torch.h"

// Standard library includes
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace fs = std::filesystem;

namespace ai_pass_selector {

enum class OptimizerType : int;

class BaseActorCritic : public AbstractAgent {
protected:
  /// Attributes on configuration
  OptimizerType actor_optimizer_type{};
  OptimizerType critic_optimizer_type{};
  double actor_learning_rate = 0.0;
  double critic_learning_rate = 0.0;
  torch::Device device = torch::kCPU;
  double discount_factor = 0.0;
  double gae_hyperparameter = 0.0;
  double entropy_coefficient = 0.0;

  /// Global Attributes
  torch::nn::Sequential actor = nullptr;
  torch::nn::Sequential critic = nullptr;
  std::shared_ptr<torch::optim::Optimizer> actor_optimizer = nullptr;
  std::shared_ptr<torch::optim::Optimizer> critic_optimizer = nullptr;

  /// Mutex for thread safety
  std::unique_ptr<std::mutex> model_mutex = std::make_unique<std::mutex>();

public:
  /// Constructors
  explicit BaseActorCritic(unsigned int max_qubits);

  /**
   *
   * @param actor
   * @param critic
   * @return
   */
  virtual bool initialize(const torch::nn::Sequential &actor,
                          const torch::nn::Sequential &critic);

  /// Destructor
  ~BaseActorCritic() override = default;

  /// Copy and move constructors and assignment operators
  BaseActorCritic(const BaseActorCritic &other) noexcept = delete;

  BaseActorCritic(BaseActorCritic &&other) noexcept = default;

  BaseActorCritic &operator=(const BaseActorCritic &other) noexcept = delete;

  BaseActorCritic &operator=(BaseActorCritic &&other) noexcept = default;

  /// Diagnostics
  void check_params(double tiny = 1e-12, double big = 1e6) const;

  //////////////////////////////////////////////////////////////////////////////
  /// Standard Actor-Critic methods

  /**
   * @param observation
   * @return [action_probs, state_value]
   */
  std::pair<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &observation);

  /**
   * Critic-only pass for bootstrapping.
   * @param observation
   * @return state_value
   */
  torch::Tensor get_value(const torch::Tensor &observation);

  /**
   *
   * @param observation
   * @return [action, log_action_probs, state_value, entropy]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  select_action(const torch::Tensor &observation);

  /**
   *
   * @param observation
   * @return
   */
  unsigned int select_greedy_action(const torch::Tensor &observation);

  /**
   * Computes advantages using Generalized Advantage Estimation.
   * @param rewards
   * @param state_values
   * @return
   */
  torch::Tensor compute_advantages(const torch::Tensor &rewards,
  const torch::Tensor &state_values);

  /**
   * Computes rewards-to-go:
   * G_t = gamma^(-t) * sum_{t'=t}^{T} gamma^(t') * R_{t'}
   * @param rewards
   * @return
   */
  torch::Tensor compute_rewards_to_go(const torch::Tensor &rewards);

  /**
   *
   * @param actor_loss
   * @param critic_loss
   */
  virtual void update_parameters(const torch::Tensor &actor_loss,
                                 const torch::Tensor &critic_loss) const;
  //////////////////////////////////////////////////////////////////////////////
  /// Saving and Loading
  virtual void save_model() const;
  void load_model() override;
};
} // namespace ai_pass_selector

#endif // BASE_ACTOR_CRITIC_HPP
