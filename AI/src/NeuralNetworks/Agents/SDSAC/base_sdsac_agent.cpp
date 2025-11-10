#include "NeuralNetworks/Agents/SDSAC/base_sdsac_agent.hpp"

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

bool BaseSDSACAgent::initialize(const torch::nn::Sequential &actor,
                                const torch::nn::Sequential &critic) {
  throw std::runtime_error(
      "BaseSDSACAgent has 4 critics and must be initialized "
      "using the 4-ciritc initialization method, not the "
      "standard 1-critic initialization method.");
}
void BaseSDSACAgent::update_parameters(const torch::Tensor &actor_loss,
                                       const torch::Tensor &critic_loss) const {
  throw std::runtime_error(
      "BaseSDSACAgent has 4 critics and must be updated using the 4-critic "
      "update method, not the standard 1-critic update method.");
}
std::pair<torch::Tensor, torch::Tensor>
BaseSDSACAgent::forward(const torch::Tensor &observation) {
  throw std::runtime_error(
      "BaseSDSACAgent does not implement the standard Actor-Critic forward "
      "method with 2 return tensors. Use sac_forward instead.");
}
torch::Tensor BaseSDSACAgent::get_value(const torch::Tensor &observation) {
  throw std::runtime_error(
      "BaseSDSACAgent does not implement the standard Actor-Critic get_value "
      "method. Use sac_get_values instead.");
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseSDSACAgent::select_action(const torch::Tensor &observation) {
  throw std::runtime_error(
      "BaseSDSACAgent does not implement the standard Actor-Critic "
      "select_action "
      "method with 4 return tensors. Use sac_select_action instead.");
}

BaseSDSACAgent::BaseSDSACAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {
  this->sdsac_temperature_alpha =
      torch::tensor(GLOBAL_PARAMS["sdsac_temperature_alpha"].to_double())
          .to(torch::kFloat32)
          .to(this->device);
  sdsac_temperature_alpha.set_requires_grad(true);
  this->sdsac_shared_learning_rate =
      GLOBAL_PARAMS["sdsac_shared_learning_rate"].to_double();
  this->sdsac_smoothing_tau = GLOBAL_PARAMS["sdsac_smoothing_tau"].to_double();
  this->sdsac_penalty_beta = GLOBAL_PARAMS["sdsac_penalty_beta"].to_double();
  this->sdsac_clip_c = GLOBAL_PARAMS["sdsac_clip_c"].to_double();
  this->sdsac_entropy_target_weight =
      GLOBAL_PARAMS["sdsac_entropy_target_weight"].to_double();
}

bool BaseSDSACAgent::initialize(const torch::nn::Sequential &actor,
                                const torch::nn::Sequential &critic_Q1_main,
                                const torch::nn::Sequential &critic_Q2_main,
                                const torch::nn::Sequential &critic_Q1_avg,
                                const torch::nn::Sequential &critic_Q2_avg) {
  try {
    this->actor = actor;
    this->critic = critic_Q1_main;
    this->critic_Q2_main = critic_Q2_main;
    this->critic_Q1_avg = critic_Q1_avg;
    this->critic_Q2_avg = critic_Q2_avg;
    // Load the model before putting it to the device to avoid device
    // scheduling issus.
    this->load_model();
    this->register_module("actor", this->actor);
    this->register_module("critic", this->critic);
    this->register_module("critic_Q2_main", this->critic_Q2_main);
    this->register_module("critic_Q1_avg", this->critic_Q1_avg);
    this->register_module("critic_Q2_avg", this->critic_Q2_avg);
    this->actor->to(this->device);
    // The main critic is critic_V_main.
    this->critic->to(this->device);
    this->critic_Q2_main->to(this->device);
    this->critic_Q1_avg->to(this->device);
    this->critic_Q2_avg->to(this->device);
    this->actor_optimizer = std::shared_ptr(
        std::move(makeOptimizer(this->actor_optimizer_type, this->actor,
                                this->sdsac_shared_learning_rate)));
    this->critic_Q1_optimizer = std::shared_ptr(
        std::move(makeOptimizer(this->critic_optimizer_type, this->critic,
                                this->sdsac_shared_learning_rate)));
    this->critic_Q2_optimizer = std::shared_ptr(std::move(
        makeOptimizer(this->critic_optimizer_type, this->critic_Q2_main,
                      this->sdsac_shared_learning_rate)));
    this->alpha_optimizer =
        std::make_shared<torch::optim::Adam>(torch::optim::Adam(
            /*params=*/{this->sdsac_temperature_alpha},
            /*defaults=*/torch::optim::AdamOptions(
                this->sdsac_shared_learning_rate)));
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
BaseSDSACAgent::sdsac_all_Q_forward(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(torch::kFloat32).to(this->device);
  return {this->actor->forward(x), this->critic->forward(x),
          this->critic_Q2_main->forward(x), this->critic_Q1_avg->forward(x),
          this->critic_Q2_avg->forward(x)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
BaseSDSACAgent::sdsac_Q_avg_only_forward(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(torch::kFloat32).to(this->device);
  return {this->actor->forward(x), this->critic_Q1_avg->forward(x),
          this->critic_Q2_avg->forward(x)};
}

std::pair<torch::Tensor, torch::Tensor>
BaseSDSACAgent::sdsac_select_action(const torch::Tensor &observation) {
  auto x = observation.to(this->device).to(torch::kFloat32);
  auto action_probs = this->actor->forward(x);
  return {
      action_probs.multinomial(/*num_samples=*/1).squeeze(-1), // Shape []
      -(action_probs * action_probs.log())
           .sum(/*dim=*/-1)
           .squeeze(-1) // Shape []
  };
}

/// [actor_loss, critic_Q1_loss, critic_Q2_loss,
/// optional_temperature_alpha_loss]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseSDSACAgent::get_loss(
    const torch::Tensor &reward,            // Shape []
    const torch::Tensor &action_probs_next, // Shape [NR_PASSES]
    const torch::Tensor &Q1_avg_next,       // Shape [NR_PASSES]
    const torch::Tensor &Q2_avg_next,       // Shape [NR_PASSES]
    const torch::Tensor &Q1_main,           // Shape [NR_PASSES]
    const torch::Tensor &Q2_main,           // Shape [NR_PASSES]
    const torch::Tensor &Q1_avg,            // Shape [NR_PASSES]
    const torch::Tensor &Q2_avg,            // Shape [NR_PASSES]
    const torch::Tensor &action,            // Shape []
    const torch::Tensor &action_probs,      // Shape [NR_PASSES]
    const torch::Tensor &old_entropy,       // Shape []
    const torch::Tensor &new_entropy        // Shape []
) {
  int32_t action_index = action.item<int32_t>();
  torch::Tensor log_action_probs_next = action_probs_next.log().detach();
  torch::Tensor expectation_Q1 = action_probs_next.dot(
      Q1_avg_next - this->sdsac_temperature_alpha * log_action_probs_next);
  torch::Tensor expectation_Q2 = action_probs_next.dot(
      Q2_avg_next - this->sdsac_temperature_alpha * log_action_probs_next);
  torch::Tensor y = reward + this->discount_factor * 0.5 *
                                 (expectation_Q1 + expectation_Q2).detach();
  torch::Tensor clip_value_Q1 =
      torch::clamp(/*self=*/Q1_main[action_index] - Q1_avg[action_index],
                   /*min=*/-this->sdsac_clip_c, /*max=*/this->sdsac_clip_c);
  torch::Tensor clip_value_Q2 =
      torch::clamp(/*self=*/Q2_main[action_index] - Q2_avg[action_index],
                   /*min=*/-this->sdsac_clip_c, /*max=*/this->sdsac_clip_c);
  torch::Tensor log_action_probs = action_probs.log();
  torch::Tensor target_entropy = this->sdsac_entropy_target_weight *
                                 torch::tensor(action_probs.size(0)).log();
  return {/*actor_loss=*/action_probs.dot(
              this->sdsac_temperature_alpha * log_action_probs -
              torch::min(Q1_main, Q2_main).detach()) +
              0.5 * this->sdsac_penalty_beta *
                  (old_entropy - new_entropy).pow(2),
          /*critic_Q1_loss=*/
          torch::max((Q1_main[action_index] - y).pow(2),
                     (Q1_avg[action_index] + clip_value_Q1 - y).pow(2)),
          /*critic_Q1_loss=*/
          torch::max((Q2_main[action_index] - y).pow(2),
                     (Q2_avg[action_index] + clip_value_Q2 - y).pow(2)),
          /*alpha_loss=*/-this->sdsac_temperature_alpha *
              action_probs.dot(log_action_probs + target_entropy).detach()};
}

void BaseSDSACAgent::sdsac_update_parameters(
    const torch::Tensor &actor_loss, const torch::Tensor &critic_Q1_loss,
    const torch::Tensor &critic_Q2_loss,
    const torch::Tensor &temperature_alpha_loss) const {
  std::lock_guard lock(*this->model_mutex);
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
  this->critic_Q1_optimizer->zero_grad();
  critic_Q1_loss.backward();
  this->critic_Q1_optimizer->step();
  this->critic_Q2_optimizer->zero_grad();
  critic_Q2_loss.backward();
  this->critic_Q2_optimizer->step();
  {
    torch::NoGradGuard no_grad_guard;
    for (const auto &pair : this->critic->named_parameters(/*recurse=*/true)) {
      const std::string &name = pair.key();
      torch::Tensor param_main = pair.value();
      torch::Tensor param_avg =
          this->critic_Q1_avg->named_parameters(/*recurse=*/true)[name];
      param_avg.mul_(1.0 - this->sdsac_smoothing_tau);
      param_avg.add_(this->sdsac_smoothing_tau * param_main);
    }
    for (const auto &pair :
         this->critic_Q2_main->named_parameters(/*recurse=*/true)) {
      const std::string &name = pair.key();
      torch::Tensor param_main = pair.value();
      torch::Tensor param_avg =
          this->critic_Q2_avg->named_parameters(/*recurse=*/true)[name];
      param_avg.mul_(1.0 - this->sdsac_smoothing_tau);
      param_avg.add_(this->sdsac_smoothing_tau * param_main);
    }
  }
  this->alpha_optimizer->zero_grad();
  temperature_alpha_loss.backward();
  this->alpha_optimizer->step();
}
} // namespace ai_pass_selector