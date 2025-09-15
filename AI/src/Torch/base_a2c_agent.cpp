#include "Torch/base_a2c_agent.hpp"

// Torch includes
#include <torch/torch.h>

// Utils includes
#include "Torch/agent_utils..hpp"

// Standard library includes

#include <utility>
#include <tuple>
#include <cmath>
#include <memory>

namespace ai_pass_selector {


BaseA2CAgent::BaseA2CAgent(
    const int max_qubits,
    const int max_instructions,
    const int max_depth,
    int nr_input_values,
    int nr_output_values,
    const torch::nn::Sequential &critic,
    const torch::nn::Sequential &actor,
    int critic_optimizer_type,
    int actor_optimizer_type,
    const double critic_learning_rate,
    const double actor_learning_rate,
    const int nr_parallel_environments,
    const torch::Device device
    ) : max_qubits(max_qubits),
        max_instructions(max_instructions),
        max_depth(max_depth),
        nr_input_values(nr_input_values),
        nr_output_values(nr_output_values),
        critic_learning_rate(critic_learning_rate),
        actor_learning_rate(actor_learning_rate),
        nr_parallel_environments(nr_parallel_environments),
        critic(std::move(critic)),
        actor(std::move(actor)),
        device(device) {
  register_module("critic", this->critic);
  register_module("actor", this->actor);
  this->critic->to(this->device);
  this->actor->to(this->device);
  this->critic_optimizer = makeOptimizer(
      critic_optimizer_type, this->critic, this->critic_learning_rate);
  this->actor_optimizer = makeOptimizer(
      actor_optimizer_type, this->actor, this->actor_learning_rate);
}


std::pair<torch::Tensor, torch::Tensor> BaseA2CAgent::forward(
    torch::Tensor batched_observations) {
  batched_observations = batched_observations.to(this->device);
  return {this->critic->forward(batched_observations),
          this->actor->forward(batched_observations)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseA2CAgent::select_action(const torch::Tensor &batched_observations) {
  auto [state_values, action_probs] = this->forward(batched_observations);
  // sample one action per row; result is [B,1] -> squeeze to [B]
  torch::Tensor actions = action_probs.multinomial(1).squeeze(-1);
  // log π(a|s) for the sampled actions: gather along the action dim
  const torch::Tensor log_action_probs = action_probs.log();
  // a_t, log π(a_t|s_t), V(s_t), entropy of π(a_t|s_t)
  return {
      actions,
      log_action_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1),
      state_values.squeeze(-1),
      -(action_probs * log_action_probs).sum(-1)
  };
}

std::pair<torch::Tensor, torch::Tensor> BaseA2CAgent::get_losses(
    const torch::Tensor &rewards,
    const torch::Tensor &log_action_probs,
    const torch::Tensor &state_values,
    const torch::Tensor &entropy,
    const torch::Tensor &termination_masks,
    const double discount_factor,
    const double gae_hyperparameter,
    const double entropy_coefficient) {

  // Let T = final timestep of an episode.
  // An episode generates T+1 states from S_0 to S_T.
  // An episode generates T actions from A_0 to A_{T-1}.
  // An episode generates T rewards from R_1 to R_T.
  int T = rewards.size(0);
  int B = rewards.size(1);
  const torch::TensorOptions options = rewards.options();
  torch::Tensor advantages = torch::zeros({T, B}, options);

  // Compute the advantages using Generalized Advantage Estimation.
  // Temporal Difference is a method used in Reinforcement Learning to estimate the value function of a state
  // based on the difference between the immediate reward obtained from a current state
  // and the estimated value of the next state.
  torch::Tensor A_gae = torch::zeros({B}, options);
  for (int t = T - 2; t >= 0; t--) {

    // In Barto & Sutton the temporal difference residual of V with discount gamma is:
    // delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    torch::Tensor delta_t
        = rewards[t] - state_values[t]
          + discount_factor * state_values[t + 1] * termination_masks[t];

    // The generalized advantage estimation defined by Schulman et al is:
    // A_gae = sum_{l=0}^{\infty} (gamma * lamda)^l * delta_{t+l}
    A_gae = discount_factor * gae_hyperparameter * A_gae * termination_masks[t]
            + delta_t;
    advantages[t] = A_gae;
  }

  // The equation for the Value function performance measure is:
  // J(w) = (1/N) * sum_{t=0}^{N-1} (A(s_t, a_t)^2)
  auto critic_loss = advantages.pow(2).mean();

  // Give a bonus for higher entropy to encourage exploration.
  // The equation for the policy performance measure is:
  // J(θ) = (1/N) * sum_{t=0}^{N-1} (ln π_θ(a_t|s_t) * A(s_t, a_t))
  auto actor_loss = -(log_action_probs * advantages.detach()).mean()
                    - entropy_coefficient * entropy.mean();
  return {critic_loss, actor_loss};
}

void BaseA2CAgent::update_parameters(const torch::Tensor &critic_loss,
                                     const torch::Tensor &actor_loss) const {
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
}
} // ai_pass_selector