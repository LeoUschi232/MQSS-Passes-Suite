#include "Agents/A2C/base_a2c_agent.hpp"

// Torch includes
#include "Agents/agent_utils.hpp"
#include "torch/torch.h"

// Standard library includes
#include <cmath>
#include <memory>
#include <tuple>
#include <utility>

namespace ai_pass_selector {

BaseA2CAgent::BaseA2CAgent(
    unsigned int max_qubits,
    std::unordered_map<std::string, std::string> params) {
  this->configure(max_qubits, std::move(params));
}

void BaseA2CAgent::configure(
    unsigned int max_qubits,
    std::unordered_map<std::string, std::string> params) {
  this->max_qubits = max_qubits;
  this->critic_optimizer_type =
      OPTIMIZER_NAME_TO_TYPE.at(params["critic_optimizer"]);
  this->actor_optimizer_type =
      OPTIMIZER_NAME_TO_TYPE.at(params["actor_optimizer"]);
  this->critic_learning_rate = std::stod(params["critic_learning_rate"]);
  this->actor_learning_rate = std::stod(params["actor_learning_rate"]);
  this->nr_parallel_environments =
      std::stoul(params["nr_parallel_environments"]);
  this->device = (params["device"] == "cuda" || params["device"] == "gpu") &&
                         torch::cuda::is_available()
                     ? torch::kCUDA
                     : torch::kCPU;
}

bool BaseA2CAgent::initialize(const torch::nn::Sequential &actor,
                              const torch::nn::Sequential &critic) {
  try {
    this->actor = actor;
    this->critic = critic;
    // Load the model before putting it to the device to avoid device
    // scheduling issus.
    this->load_model();
    register_module("critic", this->critic);
    register_module("actor", this->actor);
    this->critic->to(this->device);
    this->actor->to(this->device);
    this->critic_optimizer = makeOptimizer(critic_optimizer_type, this->critic,
                                           this->critic_learning_rate);
    this->actor_optimizer = makeOptimizer(actor_optimizer_type, this->actor,
                                          this->actor_learning_rate);
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}

unsigned int BaseA2CAgent::getMaxQubits() const { return this->max_qubits; }

std::pair<torch::Tensor, torch::Tensor>
BaseA2CAgent::forward(const torch::Tensor &batched_observations,
                      const torch::Tensor &mask) {
  std::lock_guard lock(*this->model_mutex);
  torch::Tensor x = batched_observations.to(this->device).to(torch::kFloat);
  // Do NOT reshape/flatten here.
  // Let the models handle shapes.
  return {this->actor->forward(x), this->critic->forward(x)};
}

torch::Tensor BaseA2CAgent::get_value(const torch::Tensor &batched_observations,
                                      const torch::Tensor &mask) {
  std::lock_guard lock(*this->model_mutex);
  return this->critic->forward(
      batched_observations.to(this->device).to(torch::kFloat),
      mask.to(this->device).to(torch::kBool));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseA2CAgent::select_action(const torch::Tensor &batched_observations,
                            const torch::Tensor &mask) {
  auto [action_probs, state_values] = this->forward(batched_observations, mask);

  if (action_probs.lt(0).any().item<bool>()) {
    std::cerr << "Error: action_probs contains negative values: "
              << action_probs << std::endl;
    throw std::runtime_error("Negative probabilities detected.");
  }
  if (action_probs.isnan().any().item<bool>()) {
    std::cerr << "Error: action_probs contains NaN: " << action_probs
              << std::endl;
    throw std::runtime_error("NaN in probabilities.");
  }
  if (action_probs.isinf().any().item<bool>()) {
    std::cerr << "Error: action_probs contains Inf: " << action_probs
              << std::endl;
    throw std::runtime_error("Inf in probabilities.");
  }

  // Multinomial selects num_samples=1 indices per row for the given matrix,
  // using the values in the row as weights.
  // action_probs ~ [B, NR_PASSES]
  // X.multinomial(num_samples=1) ~ [B, 1]
  // X.squeeze(dim=-1) ~ [B]
  const torch::Tensor action_indexes_unsqueezed =
      action_probs.multinomial(/*num_samples=*/1);
  const torch::Tensor action_indexes = action_indexes.squeeze(-1);

  // For advantage compute log π(a_t|s_t) for the sampled actions.
  // Gather extracts the values at specified indexes along the specified axis.
  // Parameter indexes must have the same nr of axes as the input tensor, here
  // each has 2 axes.
  // log_action_probs ~ [B, NR_PASSES]
  // X.gather(dim=-1, indexes=action_indexes_unsqueezed) ~ [B, 1]
  // X.squeeze(dim=-1) ~ [B]
  const torch::Tensor log_action_probs = action_probs.log();
  const torch::Tensor squeezed_log_action_probs =
      log_action_probs.gather(/*dim=*/-1, /*indexes=*/action_indexes_unsqueezed)
          .squeeze(-1);

  // Entropy formula H = -sum_{x}(p(x)*log(p(x)))
  // action_probs * log_action_probs ~ [B, NR_PASSES]
  // -X.sum(dim=-1) ~ [B]
  const torch::Tensor entropy =
      -(action_probs * log_action_probs).sum(/*dim=*/-1);
  return {
      action_indexes,            // Shape [B]
      squeezed_log_action_probs, // Shape [B]
      state_values,              // Shape [B]
      entropy                    // Shape [B]
  };
}

std::pair<torch::Tensor, torch::Tensor>
BaseA2CAgent::get_losses(const torch::Tensor &rewards,          // Shape [T, B]
                         const torch::Tensor &log_action_probs, // Shape [T, B]
                         const torch::Tensor &state_values, // Shape [T+1, B]
                         const torch::Tensor &entropy,      // Shape [T, B]
                         const torch::Tensor &termination_masks, // Shape [T, B]
                         const double discount_factor,
                         const double gae_hyperparameter,
                         const double entropy_coefficient) {

  // Let T = final timestep of an episode.
  // An episode generates T+1 states from S_0 to S_T.
  // An episode generates T actions from A_1 to A_T.
  // An episode generates T rewards from R_1 to R_T.
  int T = rewards.size(0);
  int B = rewards.size(1);
  const torch::TensorOptions options = rewards.options();
  torch::Tensor advantages = torch::zeros({T, B}, options);

  // Compute the advantages using Generalized Advantage Estimation.
  // Temporal Difference is a method used in Reinforcement Learning to estimate
  // the value function of a state based on the difference between the immediate
  // reward obtained from a current state and the estimated value of the next
  // state.
  torch::Tensor A_gae = torch::zeros({B}, options);
  for (int t = T - 1; t >= 0; t--) {

    // Temporal Difference Error of V(s) with discount gamma is:
    // delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    // Barto & Sutton Reinforcement Learning page 121, equation (6.5)
    torch::Tensor delta_t =
        rewards[t] - state_values[t] +
        discount_factor * state_values[t + 1] * termination_masks[t];

    // The generalized advantage estimation defined by Schulman et al is:
    // A_gae = sum_{l=0}^{\infty} (gamma * lamda)^l * delta_{t+l}
    A_gae =
        discount_factor * gae_hyperparameter * A_gae * termination_masks[t] +
        delta_t;
    advantages[t] = A_gae;
  }

  // The equation for the Value function performance measure is:
  // J(w) = (1/N) * sum_{t=0}^{N-1} (A(s_t, a_t)^2)
  auto critic_loss = advantages.pow(2).mean();

  // Give a bonus for higher entropy to encourage exploration.
  // The equation for the policy performance measure is:
  // J(θ) = (1/N) * sum_{t=0}^{N-1} (ln π_θ(a_t|s_t) * A(s_t, a_t))
  auto actor_loss = -(log_action_probs * advantages.detach()).mean() -
                    entropy_coefficient * entropy.mean();
  return {actor_loss, critic_loss};
}

void BaseA2CAgent::update_parameters(const torch::Tensor &critic_loss,
                                     const torch::Tensor &actor_loss) const {
  std::lock_guard lock(*this->model_mutex);
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
}

void BaseA2CAgent::save_model() const {
  std::lock_guard lock(*this->model_mutex);
  std::string name = this->agentName();
  if (name.empty()) {
    std::cerr << "No agent to save." << std::endl;
    return;
  }
  fs::path critic_path = fs::path(AI_AGENTS_DIR) / (name + "-critic.pt");
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  torch::save(this->critic, critic_path.string());
  torch::save(this->actor, actor_path.string());
}

void BaseA2CAgent::load_model() {
  // Silently don't load if model doesn't exist.
  std::lock_guard lock(*this->model_mutex);
  std::string name = this->agentName();
  if (name.empty()) {
    return;
  }
  fs::path critic_path = fs::path(AI_AGENTS_DIR) / (name + "-critic.pt");
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  if (!fs::exists(critic_path) || !fs::exists(actor_path)) {
    return;
  }
  torch::load(this->critic, critic_path.string(), this->device);
  torch::load(this->actor, actor_path.string(), this->device);
  std::cout << "Loaded model: " << name << std::endl;
}

} // namespace ai_pass_selector