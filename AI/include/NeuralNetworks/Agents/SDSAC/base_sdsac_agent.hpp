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
  double sdsac_entropy_target_weight = 0.0;

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

  /**
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

  //////////////////////////////////////////////////////////////////////////////
  /// SDSAC quadruple-critic methods

  /**
   * @param observation
   * @return [policy, Q1_main, Q2_main, Q1_avg, Q2_avg]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
  forward(const torch::Tensor &observation);

  /**
   * @param observation
   * @return [policy, Q1_avg, Q2_avg]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  forward_only_Q_avg(const torch::Tensor &observation);

  /**
   * @param observation
   * @return [action_index, entropy]
   */
  std::pair<torch::Tensor, torch::Tensor>
  select_action(const torch::Tensor &observation);

  /**
   * @return [actor_loss, critic_Q1_loss, critic_Q2_loss,
   * optional_temperature_alpha_loss]
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  get_loss(const torch::Tensor &reward,            // Shape []
           const torch::Tensor &action_probs_next, // Shape [NR_PASSES]
           const torch::Tensor &Q1_avg_next,       // Shape [NR_PASSES]
           const torch::Tensor &Q2_avg_next,       // Shape [NR_PASSES]
           const torch::Tensor &Q1_main,           // Shape [NR_PASSES]
           const torch::Tensor &Q2_main,           // Shape [NR_PASSES]
           const torch::Tensor &Q1_avg,            // Shape [NR_PASSES]
           const torch::Tensor &Q2_avg,            // Shape [NR_PASSES]
           const torch::Tensor &action,            // Shape []
           const torch::Tensor &action_probs,      // Shape [NR_PASSES]
           const torch::Tensor &old_entropy,       // Shape []
           const torch::Tensor &new_entropy        // Shape []
  );

  /**
   *
   * @param actor_loss
   * @param critic_Q1_loss
   * @param critic_Q2_loss
   * @param temperature_alpha_loss
   */
  void update_parameters(const torch::Tensor &actor_loss,
                         const torch::Tensor &critic_Q1_loss,
                         const torch::Tensor &critic_Q2_loss,
                         const torch::Tensor &temperature_alpha_loss) const;
  //////////////////////////////////////////////////////////////////////////////

  /// Saving and Loading
  void save_model() const override;
  void load_model() override;
};

} // namespace ai_pass_selector

#endif // BASE_SDSAC_AGENT_HPP