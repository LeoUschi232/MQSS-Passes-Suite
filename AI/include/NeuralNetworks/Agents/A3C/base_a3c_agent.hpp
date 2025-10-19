#ifndef BASE_A3C_AGENT_HPP
#define BASE_A3C_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Neural-Networks includes
#include "NeuralNetworks/Agents/base_actor_critic.hpp"

// Standard library includes
#include <memory>

namespace fs = std::filesystem;

namespace ai_pass_selector {
enum class OptimizerType : int;

class BaseA3CAgent : public BaseActorCritic {
protected:
  /// A3C specific attributes
  bool gradients_zero = true;
  bool is_boss = true;

public:
  /// Constructors
  explicit BaseA3CAgent(unsigned int max_qubits, bool is_boss = true);

  /**
   *
   * @param actor
   * @param critic
   * @return
   */
  bool initialize(const torch::nn::Sequential &actor,
                  const torch::nn::Sequential &critic) override;

  /// Destructor
  ~BaseA3CAgent() override = default;

  /// Copy and move constructors and assignment operators
  BaseA3CAgent(const BaseA3CAgent &other) noexcept = delete;

  BaseA3CAgent(BaseA3CAgent &&other) noexcept = default;

  BaseA3CAgent &operator=(const BaseA3CAgent &other) noexcept = delete;

  BaseA3CAgent &operator=(BaseA3CAgent &&other) noexcept = delete;

  //////////////////////////////////////////////////////////////////////////////
  /// A2C/A3C standard methods
  /**
   *
   * @param actor_loss
   * @param critic_loss
   */
  void update_parameters(const torch::Tensor &actor_loss,
                         const torch::Tensor &critic_loss) const override;
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  /// A3C specific methods for worker concurrency
  void zero_grad();

  /**
   * Necessary to create worker agents for asynchronous training.
   * @return
   */
  virtual std::unique_ptr<BaseA3CAgent> clone() const = 0;

  void load_weights(BaseA3CAgent &other);
  void load_gradients(BaseA3CAgent &other);
  void update_parameters_assuming_gradients_are_loaded();
  //////////////////////////////////////////////////////////////////////////////

  /// Saving and Loading
  void save_model() const override;
};
} // namespace ai_pass_selector

#endif // BASE_A3C_AGENT_HPP
