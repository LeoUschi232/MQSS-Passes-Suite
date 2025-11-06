#ifndef BASE_SAC_AGENT_HPP
#define BASE_SAC_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Neural-Networks includes
#include "NeuralNetworks/Agents/base_actor_critic.hpp"

// Standard library includes
#include <memory>
namespace fs = std::filesystem;

namespace ai_pass_selector {
enum class OptimizerType : int;

class BaseSACAgent : public BaseActorCritic {
protected:
  /// SDSAC specific attributes
  std::optional<double> sac_temperature_alpha = std::nullopt;
  double sac_shared_learning_rate = 0.0;
  double sac_smoothing_tau = 0.0;

  /// SDSAC Additional Critics
  torch::nn::Sequential critic_Q2_main{nullptr};
  torch::nn::Sequential critic_Q1_avg{nullptr};
  torch::nn::Sequential critic_Q2_avg{nullptr};
  std::shared_ptr<torch::optim::Optimizer> critic_Q1_optimizer = nullptr;
  std::shared_ptr<torch::optim::Optimizer> critic_Q2_optimizer = nullptr;

public:
  /// Constructors
  explicit BaseSACAgent(unsigned int max_qubits);

  /// Destructor
  ~BaseSACAgent() override = default;

  /// Copy and move constructors and assignment operators
  BaseSACAgent(const BaseSACAgent &other) noexcept = delete;

  BaseSACAgent(BaseSACAgent &&other) noexcept = default;

  BaseSACAgent &operator=(const BaseSACAgent &other) noexcept = delete;

  BaseSACAgent &operator=(BaseSACAgent &&other) noexcept = delete;

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
   * @param actor_loss
   * @param critic_Q1_loss
   * @param critic_Q2_loss
   */
  void update_parameters(const torch::Tensor &actor_loss,
                         const torch::Tensor &critic_Q1_loss,
                         const torch::Tensor &critic_Q2_loss) const;

  /**
   *
   * @param observation
   * @return [policy, Q1_main, Q2_main, Q1_avg, Q2_avg]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
  sac_forward(const torch::Tensor &observation);

  /**
   *
   * @param observation
   * @return
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor, torch::Tensor>
  sac_select_action(const torch::Tensor &observation);
  //////////////////////////////////////////////////////////////////////////////
};

} // namespace ai_pass_selector

#endif // BASE_SAC_AGENT_HPP