#include "Agents/A3C/base_a3c_agent.hpp"

// Torch includes
#include "Agents/agent_utils.hpp"
#include "torch/torch.h"

// Standard library includes
#include "Utils/info_utils.hpp"

#include <cmath>
#include <memory>
#include <tuple>
#include <utility>

namespace ai_pass_selector {

BaseA3CAgent::BaseA3CAgent(unsigned int max_qubits,
                           std::unordered_map<std::string, std::string> params,
                           bool is_boss)
    : max_qubits(std::max(max_qubits, GLOBAL_MIN_NR_QUBITS)), is_boss(is_boss) {
  this->configure(std::move(params));
}

void BaseA3CAgent::configure(
    std::unordered_map<std::string, std::string> params) {
  if (this->is_boss) {
    this->params_for_cloning = {
        {"device", params["device"]},
        {"actor_optimizer", params["actor_optimizer"]},
        {"critic_optimizer", params["critic_optimizer"]},
        {"actor_learning_rate", params["actor_learning_rate"]},
        {"critic_learning_rate", params["critic_learning_rate"]}};
  }
  this->device = (params["device"] == "cuda" || params["device"] == "gpu") &&
                         torch::cuda::is_available()
                     ? torch::kCUDA
                     : torch::kCPU;
  this->actor_learning_rate = std::stod(params["actor_learning_rate"]);
  this->critic_learning_rate = std::stod(params["critic_learning_rate"]);
  this->actor_optimizer_type =
      OPTIMIZER_NAME_TO_TYPE.at(params["actor_optimizer"]);
  this->critic_optimizer_type =
      OPTIMIZER_NAME_TO_TYPE.at(params["critic_optimizer"]);
}

bool BaseA3CAgent::initialize(const torch::nn::Sequential &actor,
                              const torch::nn::Sequential &critic) {
  try {
    this->actor = actor;
    this->critic = critic;
    // Load the model before putting it to the device to avoid device
    // scheduling issus.
    if (this->is_boss) {
      this->load_model();
    }
    register_module("critic", this->critic);
    register_module("actor", this->actor);
    this->critic->to(this->device);
    this->actor->to(this->device);
    if (this->is_boss) {
      // Worker agents do not need optimizers.
      // Only boss agent needs optimizers.
      this->critic_optimizer = std::shared_ptr(
          std::move(makeOptimizer(this->critic_optimizer_type, this->critic,
                                  this->critic_learning_rate)));
      this->actor_optimizer = std::shared_ptr(std::move(makeOptimizer(
          this->actor_optimizer_type, this->actor, this->actor_learning_rate)));
    }
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}

unsigned int BaseA3CAgent::getMaxQubits() const { return this->max_qubits; }

void BaseA3CAgent::zero_grad() {
  std::lock_guard lock(*this->model_mutex);
  for (auto &parameter : this->parameters()) {
    if (parameter.grad().defined()) {
      (void)parameter.grad().zero_();
    }
  }
  this->gradients_zero = true;
}

void BaseA3CAgent::load_params(BaseA3CAgent &other) {
  // Most likely the self will be one of the worker agents and the other will
  // be  the global boss agent.
  // Step: Synchronize thread-specific parameters θ'=θ and θv'=θv from
  // Mnih et al 2016.
  std::scoped_lock lock(*this->model_mutex, *other.model_mutex);
  torch::NoGradGuard no_grad_guard;

  // --- ACTOR ---
  torch::OrderedDict<std::string, torch::Tensor> source_actor_params =
      other.actor->named_parameters(/*recurse=*/true);
  torch::OrderedDict<std::string, torch::Tensor> target_actor_params =
      this->actor->named_parameters(/*recurse=*/true);

  for (auto &key_value : target_actor_params) {
    std::string key = key_value.key();
    torch::Tensor &value = key_value.value();
    if (source_actor_params.contains(key)) {
      value.copy_(source_actor_params[key].to(this->device));
    } else {
      std::cerr << "Warning: source actor model does not contain parameter "
                << key << std::endl;
    }
  }
  torch::OrderedDict<std::string, torch::Tensor> source_actor_buffers =
      other.actor->named_buffers(/*recurse=*/true);
  torch::OrderedDict<std::string, torch::Tensor> target_actor_buffers =
      this->actor->named_buffers(/*recurse=*/true);
  for (auto &key_value : target_actor_buffers) {
    std::string key = key_value.key();
    torch::Tensor &value = key_value.value();
    if (source_actor_buffers.contains(key)) {
      value.copy_(source_actor_buffers[key].to(this->device));
    } else {
      std::cerr << "Warning: source actor model does not contain buffer " << key
                << std::endl;
    }
  }

  // --- CRITIC ---
  torch::OrderedDict<std::string, torch::Tensor> source_critic_params =
      other.critic->named_parameters(/*recurse=*/true);
  torch::OrderedDict<std::string, torch::Tensor> target_critic_params =
      this->critic->named_parameters(/*recurse=*/true);
  for (auto &key_value : target_critic_params) {
    std::string key = key_value.key();
    torch::Tensor &value = key_value.value();
    if (source_critic_params.contains(key)) {
      value.copy_(source_critic_params[key].to(this->device));
    } else {
      std::cerr << "Warning: source critic model does not contain parameter "
                << key << std::endl;
    }
  }
  torch::OrderedDict<std::string, torch::Tensor> source_critic_buffers =
      other.critic->named_buffers(/*recurse=*/true);
  torch::OrderedDict<std::string, torch::Tensor> target_critic_buffers =
      this->critic->named_buffers(/*recurse=*/true);
  for (auto &key_value : target_critic_buffers) {
    std::string key = key_value.key();
    torch::Tensor &value = key_value.value();
    if (source_critic_buffers.contains(key)) {
      value.copy_(source_critic_buffers[key].to(this->device));
    } else {
      std::cerr << "Warning: source critic model does not contain buffer "
                << key << std::endl;
    }
  }
}
void BaseA3CAgent::load_gradients(BaseA3CAgent &other) {
  // Most likely the self will be the global boss agent and the other will be
  // one of the worker agents.
  // Step: Perform asynchronous update of θ using dθ and of θv using dθv from
  // Mnih et al 2016.
  std::scoped_lock lock(*this->model_mutex, *other.model_mutex);

  // --- ACTOR ---
  torch::OrderedDict<std::string, torch::Tensor> source_actor_params =
      other.actor->named_parameters(/*recurse=*/true);
  torch::OrderedDict<std::string, torch::Tensor> target_actor_params =
      this->actor->named_parameters(/*recurse=*/true);
  for (auto &key_value : target_actor_params) {
    std::string key = key_value.key();
    torch::Tensor &value = key_value.value();

    if (!source_actor_params.contains(key)) {
      std::cerr << "Warning: source actor model does not contain parameter "
                << key << std::endl;
      continue;
    }
    torch::Tensor source_gradient = source_actor_params[key].grad();
    if (!source_gradient.defined()) {
      std::cerr << "Warning: source actor model parameter " << key
                << " does not have a gradient." << std::endl;
      continue;
    }
    torch::Tensor source_gradient_detached =
        source_gradient.detach().to(this->device);
    if (!value.grad().defined()) {
      value.mutable_grad() = source_gradient_detached.clone();
    } else {
      (void)value.mutable_grad().add_(source_gradient_detached);
    }
  }
  // --- CRITIC ---
  torch::OrderedDict<std::string, torch::Tensor> source_critic_params =
      other.critic->named_parameters(/*recurse=*/true);
  torch::OrderedDict<std::string, torch::Tensor> target_critic_params =
      this->critic->named_parameters(/*recurse=*/true);
  for (auto &key_value : target_critic_params) {
    std::string key = key_value.key();
    torch::Tensor &value = key_value.value();

    if (!source_critic_params.contains(key)) {
      std::cerr << "Warning: source critic model does not contain parameter "
                << key << std::endl;
      continue;
    }
    torch::Tensor source_gradient = source_critic_params[key].grad();
    if (!source_gradient.defined()) {
      std::cerr << "Warning: source critic model parameter " << key
                << " does not have a gradient." << std::endl;
      continue;
    }
    torch::Tensor source_gradient_detached =
        source_gradient.detach().to(this->device);
    if (!value.grad().defined()) {
      value.mutable_grad() = source_gradient_detached.clone();
    } else {
      (void)value.mutable_grad().add_(source_gradient_detached);
    }
  }
  this->gradients_zero = false;
}

std::pair<torch::Tensor, torch::Tensor>
BaseA3CAgent::forward(const torch::Tensor &batched_observations) {
  torch::Tensor x = batched_observations.to(this->device).to(torch::kFloat32);
  // Do NOT reshape/flatten here.
  // Let the models handle shapes.
  return {this->actor->forward(x), this->critic->forward(x)};
}

torch::Tensor
BaseA3CAgent::get_value(const torch::Tensor &batched_observations) {
  return this->critic->forward(
      batched_observations.to(this->device).to(torch::kFloat32));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseA3CAgent::select_action(const torch::Tensor &batched_observations) {
  auto [action_probs, state_values] = this->forward(batched_observations);

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
  const torch::Tensor action_indexes = action_indexes_unsqueezed.squeeze(-1);

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
BaseA3CAgent::get_losses(const torch::Tensor &rewards,          // Shape [T]
                         const torch::Tensor &log_action_probs, // Shape [T]
                         const torch::Tensor &state_values,     // Shape [T+1]
                         const torch::Tensor &entropy,          // Shape [T]
                         const double discount_factor,
                         const double gae_hyperparameter,
                         const double entropy_coefficient) {
  // Let T = final timestep of an episode.
  // An episode generates T+1 states from S_0 to S_T.
  // An episode generates T actions from A_1 to A_T.
  // An episode generates T rewards from R_1 to R_T.
  int T = rewards.size(0);
  const torch::TensorOptions options = rewards.options();
  torch::Tensor advantages = torch::zeros({T}, options);

  // Compute the advantages using Generalized Advantage Estimation.
  // Temporal Difference is a method used in Reinforcement Learning to estimate
  // the value function of a state based on the difference between the immediate
  // reward obtained from a current state and the estimated value of the next
  // state.
  torch::Tensor A_gae = torch::zeros({1}, options);
  for (int t = T - 1; t >= 0; t--) {

    // Temporal Difference Error of V(s) with discount gamma is:
    // delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    // Barto & Sutton Reinforcement Learning page 121, equation (6.5)
    torch::Tensor delta_t =
        rewards[t] - state_values[t] + discount_factor * state_values[t + 1];

    // The generalized advantage estimation defined by Schulman et al is:
    // A_gae = sum_{l=0}^{\infty} (gamma * lamda)^l * delta_{t+l}
    A_gae = discount_factor * gae_hyperparameter * A_gae + delta_t;
    advantages[t] = A_gae;
  }

  // Give a bonus for higher entropy to encourage exploration.
  // The equation for the policy performance measure is:
  // J(θ) = (1/N) * sum_{t=0}^{N-1} (ln π_θ(a_t|s_t) * A(s_t, a_t))
  auto actor_loss = -(log_action_probs * advantages.detach()).mean() -
                    entropy_coefficient * entropy.mean();

  // The equation for the Value function performance measure is:
  // J(w) = (1/N) * sum_{t=0}^{N-1} (A(s_t, a_t)^2)
  auto critic_loss = advantages.pow(2).mean();
  return {actor_loss, critic_loss};
}

void BaseA3CAgent::update_parameters(const torch::Tensor &actor_loss,
                                     const torch::Tensor &critic_loss) const {
  if (!this->is_boss) {
    std::cerr << "Worker agents should not update parameters." << std::endl;
    return;
  }
  std::lock_guard lock(*this->model_mutex);
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
}

void BaseA3CAgent::update_parameters_assuming_gradients_are_loaded() {
  if (!this->is_boss) {
    std::cerr << "Worker agents should not update parameters." << std::endl;
    return;
  }
  std::lock_guard lock(*this->model_mutex);
  if (this->gradients_zero) {
    // Parameters were updated by another worker since load gradients was called
    // using this worker => Nothing to do.
    return;
  }
  this->actor_optimizer->step();
  this->critic_optimizer->step();
  // Better not call this->zero_grad() because it would attempt to lock again.
  // Must zero out gradients here because other functions will not do it to
  // allow races on gradient updates.
  this->actor_optimizer->zero_grad();
  this->critic_optimizer->zero_grad();
  this->gradients_zero = true;
}

void BaseA3CAgent::save_model() const {
  if (!this->is_boss) {
    std::cerr << "Worker agents should not be saved." << std::endl;
    return;
  }
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

void BaseA3CAgent::load_model() {
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