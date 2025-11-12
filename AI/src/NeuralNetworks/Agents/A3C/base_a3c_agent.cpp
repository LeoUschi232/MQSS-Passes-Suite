#include "NeuralNetworks/Agents/A3C/base_a3c_agent.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/info_utils.hpp"
#include "Utils/passes_utils.hpp"

// Standard library includes
#include <memory>
#include <utility>

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

BaseA3CAgent::BaseA3CAgent(unsigned int max_qubits, bool is_boss)
    : BaseActorCritic(max_qubits), is_boss(is_boss) {}

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
    this->register_module("actor", this->actor);
    this->register_module("critic", this->critic);
    this->actor->to(this->device);
    this->critic->to(this->device);
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

std::pair<torch::Tensor, torch::Tensor>
BaseA3CAgent::forward(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(this->device).to(torch::kFloat32);
  // Do NOT reshape/flatten here.
  // Let the models handle shapes.
  return {this->actor->forward(x), this->critic->forward(x)};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseA3CAgent::select_action(const torch::Tensor &observation) {
  auto [action_probs, state_value] = this->forward(observation);

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

void BaseA3CAgent::zero_grad() {
  std::lock_guard lock(*this->model_mutex);
  for (auto &parameter : this->parameters()) {
    if (parameter.grad().defined()) {
      (void)parameter.grad().zero_();
    }
  }
  this->gradients_zero = true;
}

std::pair<torch::Tensor, torch::Tensor>
BaseA3CAgent::get_losses(const torch::Tensor &log_action_probs, // Shape [T]
                         const torch::Tensor &state_values,     // Shape [T+1]
                         const torch::Tensor &rewards,          // Shape [T]
                         const torch::Tensor &entropy           // Shape [T]
) {
  torch::Tensor advantages = this->compute_advantages(rewards, state_values);
  return {/*actor_loss=*/-(log_action_probs * advantages.detach()).mean() -
              this->entropy_coefficient * entropy.mean(),
          /*critic_loss=*/advantages.pow(2).mean()};
}

void BaseA3CAgent::load_weights(BaseA3CAgent &other) {
  if (this->is_boss || !other.is_boss) {
    std::cerr << "Loading weights only from boss to worker allowed."
              << std::endl;
    return;
  }
  // Step: Synchronize thread-specific parameters θ'=θ and θv'=θv from
  // Mnih et al 2016.
  // Don't have to lock worker's mutex because that one is not going to be
  // undergoing changes anyway.
  std::scoped_lock lock(*other.model_mutex);
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
  if (!this->is_boss || other.is_boss) {
    std::cerr << "Loading gradients only from worker to boss allowed."
              << std::endl;
    return;
  }
  // Step: Perform asynchronous update of θ using dθ and of θv using dθv from
  // Mnih et al 2016.
  // Don't have to lock worker's mutex because that one is not going to be
  // undergoing changes anyway.
  std::scoped_lock lock(*this->model_mutex);

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
  this->actor_optimizer->zero_grad();
  this->critic_optimizer->step();
  // Better not call this->zero_grad() because it would attempt to lock again.
  // Must zero out gradients here because other functions will not do it to
  // allow races on gradient updates.
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

} // namespace ai_pass_selector