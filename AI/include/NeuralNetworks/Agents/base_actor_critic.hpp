#ifndef BASE_ACTOR_CRITIC_HPP
#define BASE_ACTOR_CRITIC_HPP

// Agents includes
#include "NeuralNetworks/Agents/abstract_agent.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/info_utils.hpp"

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
  unsigned int max_qubits = 0u;
  OptimizerType actor_optimizer_type{};
  OptimizerType critic_optimizer_type{};
  double actor_learning_rate = 0.0;
  double critic_learning_rate = 0.0;
  torch::Device device = torch::kCPU;

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
  std::pair<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &observation);

  /**
   * Critic-only pass for bootstrapping.
   * @param observation
   * @return
   */
  torch::Tensor get_value(const torch::Tensor &observation);

  /**
   *
   * @param observation
   * @return [action, log_action_prob, state_value, entropy]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  select_action(const torch::Tensor &observation);

  /**
   * No termination masks because the tensors will not be betched and will
   * therefore only ever have the T-axis.
   * @param rewards
   * @param log_action_probs
   * @param state_values
   * @param entropy
   * @param discount_factor
   * @param gae_hyperparameter
   * @param entropy_coefficient
   * @return [actor_loss, critic_loss]
   */
  static std::pair<torch::Tensor, torch::Tensor>
  get_losses(const torch::Tensor &rewards,
             const torch::Tensor &log_action_probs,
             const torch::Tensor &state_values, const torch::Tensor &entropy,
             double discount_factor, double gae_hyperparameter,
             double entropy_coefficient);

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
