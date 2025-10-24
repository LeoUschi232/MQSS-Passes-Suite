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
  this->ppo_value_loss_coefficient =
      GLOBAL_PARAMS["ppo_value_loss_coefficient"].to_double();
  this->min_ratio = 1.0 - ppo_epsilon;
  this->max_ratio = 1.0 + ppo_epsilon;
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
torch::Tensor
BasePPOAgent::get_total_loss(const torch::Tensor &advantages,
                             const torch::Tensor &old_log_action_probs, // [T]
                             const torch::Tensor &old_state_values,     // [T+1]
                             const torch::Tensor &new_log_action_probs, // [T]
                             const torch::Tensor &new_state_values,     // [T+1]
                             const torch::Tensor &entropy               // [T]
) {
  torch::Tensor ratio = torch::exp(new_log_action_probs - old_log_action_probs);
  torch::Tensor surrogate1 = ratio * advantages;
  torch::Tensor surrogate2 =
      torch::clamp(ratio, this->min_ratio, this->max_ratio) * advantages;
  torch::Tensor policy_loss = -torch::min(surrogate1, surrogate2).mean();
  auto T = advantages.size(0);
  torch::Tensor returns =
      advantages + old_state_values.narrow(/*dim=*/0, /*start=*/0,
                                           /*length=*/T);
  torch::Tensor value_loss = (new_state_values.narrow(/*dim=*/0, /*start=*/0,
                                                      /*length=*/T) -
                              returns)
                                 .pow(2)
                                 .mean();
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