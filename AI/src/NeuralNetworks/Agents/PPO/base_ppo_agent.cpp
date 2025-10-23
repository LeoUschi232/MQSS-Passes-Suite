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
  this->ppo_epsilon = GLOBAL_PARAMS["ppo_epsilon"].to_double();
  this->ppo_value_loss_coefficient =
      GLOBAL_PARAMS["ppo_value_loss_coefficient"].to_double();
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
torch::Tensor BasePPOAgent::get_losses(
    const torch::Tensor &advantages, const torch::Tensor &old_log_action_probs,
    const torch::Tensor &old_state_values,
    const torch::Tensor &new_log_action_probs,
    const torch::Tensor &new_state_values, const torch::Tensor &entropy) {
  torch::Tensor ratios =
      torch::exp(new_log_action_probs - old_log_action_probs);
  torch::Tensor policy_loss =
      -torch::min(ratios * advantages,
                  torch::clamp(ratios, /*min=*/1.0 - this->ppo_epsilon,
                               /*max=*/1.0 + this->ppo_epsilon) *
                      advantages)
           .mean();
  torch::Tensor returns = advantages + old_state_values;
  torch::Tensor value_loss = (new_state_values - returns).pow(2).mean();
  torch::Tensor entropy_loss = entropy.mean();
  return policy_loss + this->ppo_value_loss_coefficient * value_loss -
         this->entropy_coefficient * entropy_loss;
}

void BasePPOAgent::update_parameters(const torch::Tensor &total_loss) const {
  std::lock_guard lock(*this->model_mutex);
  this->actor_optimizer->zero_grad();
  this->critic_optimizer->zero_grad();
  total_loss.backward();
  this->actor_optimizer->step();
  this->critic_optimizer->step();
}

void BasePPOAgent::save_model() const { BaseActorCritic::save_model(); }
} // namespace ai_pass_selector