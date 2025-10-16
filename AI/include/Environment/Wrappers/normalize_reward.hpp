#ifndef NORMALIZE_REWARD_HPP
#define NORMALIZE_REWARD_HPP

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

namespace ai_pass_selector {
class NormalizeReward : public QuantumCircuitEnvironment {
  double discount = 0.99;
  double epsilon = 1e-8;
  double discounted_reward = 0.0;
  double mean = 0.0;
  double variance = 0.0;

public:
  /// Constructor
  explicit NormalizeReward(std::unique_ptr<QuantumCircuitEnvironment> env) :
  QuantumCircuitEnvironment(

  /// Destructor
  ~NormalizeReward() = default;

  /// Copy and Move constructors and assignments
  NormalizeReward(const NormalizeReward &other) = delete;
  NormalizeReward &operator=(const NormalizeReward &other) = delete;
  NormalizeReward(NormalizeReward &&other) noexcept = default;
  NormalizeReward &operator=(NormalizeReward &&) noexcept = default;


};
} // namespace ai_pass_selector

#endif // NORMALIZE_REWARD_HPP
