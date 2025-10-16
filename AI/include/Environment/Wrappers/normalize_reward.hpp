#ifndef NORMALIZE_REWARD_HPP
#define NORMALIZE_REWARD_HPP

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

namespace ai_pass_selector {
class NormalizeReward : public QuantumCircuitEnvironment {
  double discount = 0.99;
  double epsilon = 1e-8;
  double discounted_reward = 0.0;
  unsigned int count = 0u;
  double mean = 0.0;
  double variance = 1.0;

public:
  /// Constructor
  explicit NormalizeReward(const QuantumCircuitEnvironment &environment)
      : QuantumCircuitEnvironment(environment.getMaxQubits()) {}

  /// Destructor
  ~NormalizeReward() override = default;

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
