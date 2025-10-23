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
   * @param initial_observations
   */
  std::pair<std::vector<torch::Tensor>, torch::Tensor>
  compute_rollout_data(const torch::Tensor &initial_observations);

  /**
   *
   * @param rollout_observations
   * @param rollout_actions
   * @param advantages
   * @param log_action_probs
   * @param state_values
   * @return
   */
  std::pair<torch::Tensor, torch::Tensor> get_losses(
      const std::vector<torch::Tensor> &rollout_observations,
      const torch::Tensor &rollout_actions, const torch::Tensor &advantages,
      const torch::Tensor &log_action_probs, const torch::Tensor &state_values);
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