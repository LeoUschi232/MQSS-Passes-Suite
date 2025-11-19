#ifndef BASE_ACTOR_CRITIC_HPP
#define BASE_ACTOR_CRITIC_HPP

// Agents includes
#include "NeuralNetworks/Agents/abstract_agent.hpp"

// Torch includes
#include "torch/torch.h"

// Standard library includes
#include <memory>
#include <mutex>

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
   * @return
   */
  virtual unsigned int select_greedy_action(const torch::Tensor &observation);

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
   * Computes the Kullback-Leibler divergence D_KL(P||Q) between two policies.
   * D_KL(P||Q) = sum_i(P(i)*(log(P(i))-log(Q(i))))
   * @param policy_p
   * @param policy_q
   * @return
   */
  torch::Tensor compute_KL_divergence(const torch::Tensor &policy_p,
                                      const torch::Tensor &policy_q) const;

  //////////////////////////////////////////////////////////////////////////////
  /**
   * @param circuit_path
   * @return
   */
  std::vector<std::function<std::unique_ptr<mlir::Pass>()>>
  select_passes_for_circuit(const fs::path &circuit_path) override;

  /// Saving and Loading
  virtual void save_model() const;
  void load_model() override;
};
} // namespace ai_pass_selector

#endif // BASE_ACTOR_CRITIC_HPP
