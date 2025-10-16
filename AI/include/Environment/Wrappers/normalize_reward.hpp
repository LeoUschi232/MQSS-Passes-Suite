#ifndef NORMALIZE_REWARD_HPP
#define NORMALIZE_REWARD_HPP

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

namespace ai_pass_selector {
class NormalizeReward {
  std::unique_ptr<QuantumCircuitEnvironment> environment_ptr;

public:
  NormalizeReward(const QuantumCircuitEnvironment &environment)
      : environment_ptr(std::unique_ptr<QuantumCircuitEnvironment>(environment)) {}
  ~NormalizeReward() = default;
};
} // namespace ai_pass_selector

#endif // NORMALIZE_REWARD_HPP
