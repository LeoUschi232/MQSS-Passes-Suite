#include "Torch/a2c_agent.hpp"

// Torch includes
#include <torch/torch.h>

// Standard library includes
#include <utility>
#include <tuple>
#include <cmath>
#include <memory>
#include <sstream>

namespace ai_pass_selector {


A2CAgent::A2CAgent(
    const int nr_input_values,
    const int nr_actions,
    const int max_qubits,
    const int max_instructions,
    const int max_depth,
    const double critic_learning_rate,
    const double actor_learning_rate,
    const int nr_parallel_environments,
    const torch::Device device
    ) : nr_input_values(nr_input_values),
        nr_actions(nr_actions),
        max_qubits(max_qubits),
        max_instructions(max_instructions),
        max_depth(max_depth),
        critic_learning_rate(critic_learning_rate),
        actor_learning_rate(actor_learning_rate),
        nr_parallel_environments(nr_parallel_environments),
        device(device) {
  int critic_layer1_size = static_cast<int>(std::lround(
      std::cbrt(static_cast<double>(nr_input_values * nr_input_values))));
  int critic_layer2_size = static_cast<int>(std::lround(
      std::cbrt(static_cast<double>(nr_input_values))));
  int actor_layer1_size = static_cast<int>(std::lround(std::cbrt(
      static_cast<double>(nr_input_values * nr_input_values * nr_actions))));
  int actor_layer2_size = static_cast<int>(std::lround(std::cbrt(
      static_cast<double>(nr_input_values * nr_actions * nr_actions))));

  this->critic = torch::nn::Sequential(
      torch::nn::Linear(nr_input_values, critic_layer1_size),
      torch::nn::LeakyReLU(),
      torch::nn::Linear(critic_layer1_size, critic_layer2_size),
      torch::nn::LeakyReLU(),
      torch::nn::Linear(critic_layer2_size, 1));
  this->critic->to(this->device);
  this->actor = torch::nn::Sequential(
      torch::nn::Linear(nr_input_values, actor_layer1_size),
      torch::nn::LeakyReLU(),
      torch::nn::Linear(actor_layer1_size, actor_layer2_size),
      torch::nn::LeakyReLU(),
      torch::nn::Linear(actor_layer2_size, nr_actions),
      torch::nn::Softmax(torch::nn::SoftmaxOptions(/*dim=*/-1)));
  this->actor->to(this->device);
  this->critic_optimizer = std::make_unique<torch::optim::Adam>(
      critic->parameters(), torch::optim::AdamOptions(critic_learning_rate));
  this->actor_optimizer = std::make_unique<torch::optim::Adam>(
      actor->parameters(), torch::optim::AdamOptions(actor_learning_rate));
}


std::pair<torch::Tensor, torch::Tensor> A2CAgent::forward(
    torch::Tensor batched_observations) {
  batched_observations = batched_observations.to(this->device);
  return {this->critic->forward(batched_observations),
          this->actor->forward(batched_observations)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
A2CAgent::select_action(const torch::Tensor &batched_observations) {
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

std::pair<torch::Tensor, torch::Tensor> A2CAgent::get_losses(
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

void A2CAgent::update_parameters(const torch::Tensor &critic_loss,
                                 const torch::Tensor &actor_loss) {
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
}

std::string
A2CAgent::make_path(const std::string &dir, const std::string &who) const {
  std::ostringstream os;
  os << dir << "/" << who << "-"
      << this->nr_input_values << "x" << this->nr_actions << ".h5";
  return os.str();
}

void A2CAgent::save_model(const std::string &weights_dir) const {
  std::string critic_path = this->make_path(weights_dir, "critic");
  torch::save(this->critic, critic_path);
  std::string actor_path = this->make_path(weights_dir, "actor");
  torch::save(this->actor, actor_path);
}

void A2CAgent::load_model(const std::string &weights_dir) {
  std::string critic_path = this->make_path(weights_dir, "critic");
  torch::load(this->critic, critic_path);
  this->critic->to(this->device);
  std::string actor_path = this->make_path(weights_dir, "actor");
  torch::load(this->actor, actor_path);
  this->actor->to(this->device);
}

} // ai_pass_selector