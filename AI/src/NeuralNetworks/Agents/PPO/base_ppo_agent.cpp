#include "NeuralNetworks/Agents/PPO/base_ppo_agent.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Utils includes
#include "Utils/info_utils.hpp"

// Standard library includes
#include <utility>

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;
BasePPOAgent::BasePPOAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {
  double ppo_epsilon = GLOBAL_PARAMS["ppo_epsilon"].to_double();
  this->min_ratio = 1.0 - ppo_epsilon;
  this->max_ratio = 1.0 + ppo_epsilon;
}

bool BasePPOAgent::initialize(const torch::nn::Sequential &actor,
                              const torch::nn::Sequential &critic) {
  try {
    this->actor = actor;
    this->critic = critic;
    // Load the model before putting it to the device to avoid device
    // scheduling issus.
    this->load_model();
    this->register_module("critic", this->critic);
    this->register_module("actor", this->actor);
    this->critic->to(this->device);
    this->actor->to(this->device);
    this->critic_optimizer = std::shared_ptr(
        std::move(makeOptimizer(this->critic_optimizer_type, this->critic,
                                this->critic_learning_rate)));
    this->actor_optimizer = std::shared_ptr(std::move(makeOptimizer(
        this->actor_optimizer_type, this->actor, this->actor_learning_rate)));
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}

std::pair<torch::Tensor, torch::Tensor>
BasePPOAgent::forward(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(this->device).to(torch::kFloat32);
  // Do NOT reshape/flatten here.
  // Let the models handle shapes.
  return {this->actor->forward(x), this->critic->forward(x)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
BasePPOAgent::force_select_action(
    const torch::Tensor &observation,
    const torch::Tensor &action_index_unsqueezed) {
  auto [action_probs, state_value] = this->forward(observation);
  const torch::Tensor log_action_probs = action_probs.log();
  const torch::Tensor squeezed_log_action_probs =
      log_action_probs.gather(/*dim=*/-1, /*indexes=*/action_index_unsqueezed)
          .squeeze(-1);
  const torch::Tensor entropy =
      -(action_probs * log_action_probs).sum(/*dim=*/-1).squeeze(-1);
  return {squeezed_log_action_probs, state_value, entropy};
}
std::pair<torch::Tensor, torch::Tensor>
BasePPOAgent::get_losses(const torch::Tensor &old_log_action_probs, // [T]
                         const torch::Tensor &old_state_values,     // [T+1]
                         const torch::Tensor &new_log_action_probs, // [T]
                         const torch::Tensor &new_state_values,     // [T+1]
                         const torch::Tensor &rewards,              // [T]
                         const torch::Tensor &entropy               // [T]
) {
  torch::Tensor ratio = torch::exp(new_log_action_probs - old_log_action_probs);
  torch::Tensor old_advantages =
      this->compute_advantages(rewards, old_state_values).detach();
  int64_t T = rewards.size(0);
  return {/*actor_loss=*/-torch::min(
              ratio * old_advantages,
              torch::clamp(ratio, this->min_ratio, this->max_ratio) *
                  old_advantages)
                  .mean() -
              this->entropy_coefficient * entropy.mean(),
          /*critic_loss=*/(
              new_state_values.narrow(/*dim=*/0, /*start=*/0, /*length=*/T) -
              old_advantages -
              old_state_values.narrow(/*dim=*/0, /*start=*/0, /*length=*/T))
              .pow(2)
              .mean()};
}
} // namespace ai_pass_selector