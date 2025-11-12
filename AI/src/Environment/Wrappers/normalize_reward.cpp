#include "Environment/Wrappers/normalize_reward.hpp"

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

NormalizeReward::NormalizeReward(const QuantumCircuitEnvironment &environment)
    : QuantumCircuitEnvironment(environment.getMaxQubits()) {
  this->discount_factor = GLOBAL_PARAMS["discount_factor"].to_double();
  this->circuit_path = environment.getCircuitPath();
  if (auto optional_randomizer_params =
          environment.getRegisteredRandomizerParams();
      optional_randomizer_params.has_value()) {
    std::tie(this->qubits_cholesky_params, this->gates_weights) =
        optional_randomizer_params.value();
  }
}

void NormalizeReward::reset(std::optional<int> seed) {
  QuantumCircuitEnvironment::reset(seed);
  this->discounted_reward = 0.0;
}

std::tuple<float, bool, bool> NormalizeReward::step(unsigned int action) {
  auto [reward, terminated, truncated] =
      QuantumCircuitEnvironment::step(action);
  if (terminated) {
    this->discounted_reward = reward;
  } else {
    this->discounted_reward =
        this->discount_factor * this->discounted_reward + reward;
  }
  // n_t = n_{t-1} + 1
  this->count += 1.0f;
  // 1/n_t
  float nt = 1.0 / this->count;
  // 1 - 1/n_t
  float difference1 = 1.0 - nt;
  // G_t - mu_{t-1}
  float difference2 = this->discounted_reward - this->mean;
  this->mean += nt * difference2;
  this->variance =
      difference1 * (this->variance + nt * difference2 * difference2);
  return {reward / std::sqrt(this->variance + this->epsilon), terminated,
          truncated};
}

} // namespace ai_pass_selector
