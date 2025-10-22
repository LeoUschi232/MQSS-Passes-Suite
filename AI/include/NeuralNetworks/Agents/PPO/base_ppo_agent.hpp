#ifndef BASE_PPO_AGENT_HPP
#define BASE_PPO_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Neural-Networks includes
#include "NeuralNetworks/Agents/base_actor_critic.hpp"

// Standard library includes
#include <memory>

namespace fs = std::filesystem;

namespace ai_pass_selector {
enum class OptimizerType : int;

class BasePPOAgent : public BaseActorCritic {
protected:
  /// PPO specific attributes
  double ppo_epsilon = 0.2;
  torch::Tensor rollout_observations;
  torch::Tensor rollout_actions;

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
  ///
  /**
   * @param observations
   * @param actions
   */
  void set_rollout_data(const torch::Tensor &observations,
                        const torch::Tensor &actions);

  /**
   *
   * @param rewards
   * @param log_action_probs
   * @param state_values
   * @param entropy
   * @param discount_factor
   * @param gae_hyperparameter
   * @param entropy_coefficient
   * @return
   */
  std::pair<torch::Tensor, torch::Tensor>
  get_losses(const torch::Tensor &rewards,
             const torch::Tensor &log_action_probs,
             const torch::Tensor &state_values, const torch::Tensor &entropy,
             double discount_factor, double gae_hyperparameter,
             double entropy_coefficient) override;
  //////////////////////////////////////////////////////////////////////////////

  /**
   *
   * @param circuit_path
   * @return
   */
  std::vector<std::function<std::unique_ptr<mlir::Pass>()>>
  select_passes_for_circuit(const fs::path &circuit_path) override;
  /// Saving and Loading
  void save_model() const override;
};
} // namespace ai_pass_selector

#endif // BASE_PPO_AGENT_HPP