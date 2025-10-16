#include "Environment/Wrappers/normalize_reward.hpp"

namespace ai_pass_selector {

std::tuple<float, bool, bool> NormalizeReward::step(unsigned int action) {
  auto [reward, terminated, truncated] =
      QuantumCircuitEnvironment::step(action);
  if (terminated) {
    this->discounted_reward = reward;
  } else {
    this->discounted_reward = this->discount * this->discounted_reward + reward;
  }
  // n_t = n_{t-1} + 1
  this->count++;
  // 1/n_t
  double nt = 1.0 / static_cast<double>(this->count);
  // 1 - 1/n_t
  double difference1 = 1.0 - nt;
  // G_t - mu_{t-1}
  double difference2 = this->discounted_reward - this->mean;
  this->mean += nt * difference2;
  this->variance =
      difference1 * (this->variance + nt * difference2 * difference2);
  return {
      static_cast<float>(reward / std::sqrt(this->variance + this->epsilon)),
      terminated, truncated};
}

} // namespace ai_pass_selector