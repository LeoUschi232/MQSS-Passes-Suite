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
  double acer_retrace_clip_c = 0.0;
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

  /**
   *
   * @param actor
   * @param critic
   * @return
   */
  bool initialize(const torch::nn::Sequential &actor,
                  const torch::nn::Sequential &critic) override;

  //////////////////////////////////////////////////////////////////////////////
  /// ACER standard methods

  //////////////////////////////////////////////////////////////////////////////
  /// Saving and Loading
  void save_model() const override;
};
} // namespace ai_pass_selector

#endif // BASE_ACER_AGENT_HPP
