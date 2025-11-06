#ifndef NORMALIZE_REWARD_HPP
#define NORMALIZE_REWARD_HPP

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

namespace ai_pass_selector {
class NormalizeReward : public QuantumCircuitEnvironment {
  /// Attributes
  float discount_factor = 0.995f;
  float discounted_reward = 0.0f;
  float mean = 0.0f;
  float variance = 1.0f;
  // Set the count to 1 at the beginning to fake having had an episode with
  // reward 0.0, to make all episodes compute their variance against that.
  // If this isn't done, the first episode will have a variance of about 0.0f
  // and will blow up the reward.
  float count = 1.0f;
  float epsilon = 1e-8f;

public:
  /// Constructor
  explicit NormalizeReward(const QuantumCircuitEnvironment &environment);

  /// Destructor
  ~NormalizeReward() override = default;

  /// Reset environment and reward normalization state
  void reset() override;

  /// Copy and Move constructors and assignments
  NormalizeReward(const NormalizeReward &other) = delete;
  NormalizeReward &operator=(const NormalizeReward &other) = delete;
  NormalizeReward(NormalizeReward &&other) noexcept = default;
  NormalizeReward &operator=(NormalizeReward &&) noexcept = default;

  /**
   *
   * @param action
   * @return
   */
  std::tuple<float, bool, bool> step(unsigned int action) override;
};
} // namespace ai_pass_selector

#endif // NORMALIZE_REWARD_HPP
