#ifndef BASE_ACER_AGENT_HPP
#define BASE_ACER_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Neural-Networks includes
#include "NeuralNetworks/Agents/base_actor_critic.hpp"

// Standard library includes
#include <memory>

namespace fs = std::filesystem;

namespace ai_pass_selector {
enum class OptimizerType : int;

class BaseACERAgent : public BaseActorCritic {
protected:
  /// ACER specific attributes
  double acer_truncation_threshold_c = 0.0;
  double acer_trust_region_delta = 0.0;

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

  //////////////////////////////////////////////////////////////////////////////
  /// ACER override for disabling methods
  bool initialize(const torch::nn::Sequential &actor,
                  const torch::nn::Sequential &critic) override;
  void update_parameters(const torch::Tensor &actor_loss,
                         const torch::Tensor &critic_loss) const override;
  std::pair<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &observation) override;
  //////////////////////////////////////////////////////////////////////////////
  /// ACER standard methods
  /**
   * @param observation
   * @return
   */
  torch::Tensor get_value(const torch::Tensor &observation) override;

  /**
   * @param observation
   * @return [action_probs, state_value]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  acer_forward(const torch::Tensor &observation);

  /**
   *
   * @param observation
   * @return
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  select_action(const torch::Tensor &observation) override;

  //////////////////////////////////////////////////////////////////////////////
  /// Saving and Loading
  void save_model() const override;
};
} // namespace ai_pass_selector

#endif // BASE_ACER_AGENT_HPP
