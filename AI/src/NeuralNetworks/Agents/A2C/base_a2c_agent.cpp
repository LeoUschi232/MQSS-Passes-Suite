#include "NeuralNetworks/Agents/A2C/base_a2c_agent.hpp"

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

BaseA2CAgent::BaseA2CAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {}

bool BaseA2CAgent::initialize(const torch::nn::Sequential &actor,
                              const torch::nn::Sequential &critic) {
  try {
    this->actor = actor;
    this->critic = critic;
    // Load the model before putting it to the device to avoid device
    // scheduling issus.
    this->load_model();
    this->register_module("actor", this->actor);
    this->register_module("critic", this->critic);
    this->actor->to(this->device);
    this->critic->to(this->device);
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
BaseA2CAgent::forward(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(this->device).to(torch::kFloat32);
  // Do NOT reshape/flatten here.
  // Let the models handle shapes.
  return {this->actor->forward(x), this->critic->forward(x)};
}

torch::Tensor BaseA2CAgent::get_value(const torch::Tensor &observation) {
  return this->critic->forward(
      observation.to(this->device).to(torch::kFloat32));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseA2CAgent::select_action(const torch::Tensor &observation) {
  auto [action_probs, state_value] = this->forward(observation);
  if ((action_probs < 0).any().item<bool>()) {
    throw std::runtime_error("A2C action_probs contains x<0.");
  }
  if (torch::isinf(action_probs).any().item<bool>()) {
    throw std::runtime_error("A2C action_probs contains Inf.");
  }
  if (torch::isnan(action_probs).any().item<bool>()) {
    throw std::runtime_error("A2C action_probs contains NaN.");
  }

  // Multinomial selects num_samples=1 indices per row for the given matrix,
  // using the values in the row as weights.
  // action_probs ~ [NR_PASSES]
  // X.multinomial(num_samples=1) ~ [1]
  // X.squeeze(dim=-1) ~ []
  const torch::Tensor action_index_unsqueezed = action_probs.multinomial(1);
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
      unsqueezed_log_action_probs.gather(-1, action_index_unsqueezed)
          .squeeze(-1);

  // Entropy formula H = -sum_{x}(p(x)*log(p(x)))
  // action_probs * log_action_probs ~ [NR_PASSES]
  // -X.sum(dim=-1) ~ [1]
  // X.squeeze(dim=-1) ~ []
  const torch::Tensor entropy =
      -(action_probs * unsqueezed_log_action_probs).sum(-1).squeeze(-1);
  return {
      action_index,    // Shape []
      log_action_prob, // Shape []
      state_value,     // Shape []
      entropy          // Shape []
  };
}

std::pair<torch::Tensor, torch::Tensor>
BaseA2CAgent::get_losses(const torch::Tensor &log_action_probs, // Shape [T]
                         const torch::Tensor &state_values,     // Shape []
                         const torch::Tensor &final_state_value,     // Shape []
                         const torch::Tensor &rewards,          // Shape [T]
                         const torch::Tensor &entropy           // Shape [T]
) {
  // Advantages are detached from all states throughout the GAE computation graph.
  // This is because critic losses should have fixed-target returns and only
  // consider values of its own timestep.
  torch::Tensor advantages = this->compute_advantages(rewards, state_values, final_state_value);
  return {/*actor_loss=*/-(log_action_probs * advantages).mean() -
              this->entropy_coefficient * entropy.mean(),
          /*critic_loss=*/(state_values - state_values.detach() - advantages)
              .pow(2)
              .mean()};
}

void BaseA2CAgent::update_parameters(const torch::Tensor &actor_loss,
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
