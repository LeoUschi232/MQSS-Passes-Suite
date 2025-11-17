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

torch::Tensor BasePPOAgent::get_value(const torch::Tensor &observation) {
  return this->critic->forward(
      observation.to(this->device).to(torch::kFloat32));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BasePPOAgent::select_action(const torch::Tensor &observation) {
  auto [action_probs, state_value] = this->forward(observation);
  if ((action_probs < 0).any().item<bool>()) {
    throw std::runtime_error("PPO action_probs contains x<0.");
  }
  if (torch::isinf(action_probs).any().item<bool>()) {
    throw std::runtime_error("PPO action_probs contains Inf.");
  }
  if (torch::isnan(action_probs).any().item<bool>()) {
    throw std::runtime_error("PPO action_probs contains NaN.");
  }

  // Multinomial selects num_samples=1 indices per row for the given matrix,
  // using the values in the row as weights.
  // action_probs ~ [NR_PASSES]
  // X.multinomial(num_samples=1) ~ [1]
  // X.squeeze(dim=-1) ~ []
  const torch::Tensor action_index_unsqueezed =
      action_probs.multinomial(/*num_samples=*/1);
  const torch::Tensor action_index = action_index_unsqueezed.squeeze(-1);

  // For advantage compute log π(a_t|s_t) for the sampled actions.
  // Gather extracts the values at specified indexes along the specified axis.
  // Parameter indexes must have the same nr of axes as the input tensor, here
  // each has 2 axes.
  // unsqueezed_log_action_probs ~ [NR_PASSES]
  // X.gather(dim=-1, indexes=action_indexes_unsqueezed) ~ [1]
  // X.squeeze(dim=-1) ~ []
  const torch::Tensor unsqueezed_log_action_probs = action_probs.log();
  const torch::Tensor log_action_prob =
      unsqueezed_log_action_probs
          .gather(/*dim=*/-1, /*indexes=*/action_index_unsqueezed)
          .squeeze(-1);

  // Entropy formula H = -sum_{x}(p(x)*log(p(x)))
  // action_probs * log_action_probs ~ [NR_PASSES]
  // -X.sum(dim=-1) ~ [1]
  // X.squeeze(dim=-1) ~ []
  const torch::Tensor entropy =
      -(action_probs * unsqueezed_log_action_probs).sum(/*dim=*/-1).squeeze(-1);
  return {
      action_index,    // Shape []
      log_action_prob, // Shape []
      state_value,     // Shape []
      entropy          // Shape []
  };
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

void BasePPOAgent::update_parameters(const torch::Tensor &actor_loss,
                                     const torch::Tensor &critic_loss) const {
  std::lock_guard lock(*this->model_mutex);
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
}
} // namespace ai_pass_selector