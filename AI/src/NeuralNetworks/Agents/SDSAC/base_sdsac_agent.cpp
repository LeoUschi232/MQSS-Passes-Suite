#include "NeuralNetworks/Agents/SDSAC/base_sdsac_agent.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"
#include "NeuralNetworks/layers_and_wrappers.hpp"

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
extern torch::TensorOptions GLOBAL_TENSOR_OPTIONS;

BaseSDSACAgent::BaseSDSACAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {
  this->sdsac_temperature_alpha =
      torch::tensor(GLOBAL_PARAMS["sdsac_temperature_alpha"].to_double())
          .to(torch::kFloat32)
          .to(this->device)
          .set_requires_grad(true);
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
    this->sdsac_temperature_alpha =
        this->sdsac_temperature_alpha.to(this->device);
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
            {this->sdsac_temperature_alpha},
            torch::optim::AdamOptions(this->sdsac_shared_learning_rate)));
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
BaseSDSACAgent::forward(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(torch::kFloat32).to(this->device);
  return {this->actor->forward(x), this->critic->forward(x),
          this->critic_Q2_main->forward(x), this->critic_Q1_avg->forward(x),
          this->critic_Q2_avg->forward(x)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
BaseSDSACAgent::forward_only_Q_avg(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(torch::kFloat32).to(this->device);
  return {this->actor->forward(x), this->critic_Q1_avg->forward(x),
          this->critic_Q2_avg->forward(x)};
}

std::pair<torch::Tensor, torch::Tensor>
BaseSDSACAgent::select_action(const torch::Tensor &observation) {
  auto x = observation.to(this->device).to(torch::kFloat32);
  auto action_probs = this->actor->forward(x);
  if ((action_probs < 0).any().item<bool>()) {
    throw std::runtime_error("SDSAC action_probs contains x<0.");
  }
  if (torch::isinf(action_probs).any().item<bool>()) {
    throw std::runtime_error("SDSAC action_probs contains Inf.");
  }
  if (torch::isnan(action_probs).any().item<bool>()) {
    throw std::runtime_error("SDSAC action_probs contains NaN.");
  }
  return {
      action_probs.multinomial(1).squeeze(-1),                 // Shape []
      -(action_probs * action_probs.log()).sum(-1).squeeze(-1) // Shape []
  };
}

/// [actor_loss, critic_Q1_loss, critic_Q2_loss, temperature_alpha_loss]
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
    const torch::Tensor &old_entropy        // Shape []
) {
  int32_t action_index = action.item<int32_t>();
  torch::Tensor alpha_log_action_probs_next =
      (this->sdsac_temperature_alpha * action_probs_next.log()).detach();
  torch::Tensor expectation_Q1 =
      action_probs_next.dot(Q1_avg_next - alpha_log_action_probs_next);
  torch::Tensor expectation_Q2 =
      action_probs_next.dot(Q2_avg_next - alpha_log_action_probs_next);
  torch::Tensor y =
      (reward +
       this->discount_factor * torch::average(expectation_Q1, expectation_Q2))
          .detach();
  torch::Tensor clip_value_Q1 =
      torch::clamp(Q1_main[action_index] - Q1_avg[action_index],
                   -this->sdsac_clip_c, this->sdsac_clip_c);
  torch::Tensor clip_value_Q2 =
      torch::clamp(Q2_main[action_index] - Q2_avg[action_index],
                   -this->sdsac_clip_c, this->sdsac_clip_c);
  torch::Tensor log_action_probs = action_probs.log();
  torch::Tensor new_entropy =
      -(action_probs * log_action_probs).sum(-1).squeeze(-1);
  torch::Tensor target_entropy =
      this->sdsac_entropy_target_weight *
      torch::tensor(action_probs.size(0), GLOBAL_TENSOR_OPTIONS).log();
  return {action_probs.dot(this->sdsac_temperature_alpha.detach() *
                               log_action_probs -
                           torch::average(Q1_main, Q2_main).detach()) +
              0.5 * this->sdsac_penalty_beta *
                  (old_entropy - new_entropy).square(),
          torch::max((Q1_main[action_index] - y).square(),
                     (Q1_avg[action_index] + clip_value_Q1 - y).square()),
          torch::max((Q2_main[action_index] - y).square(),
                     (Q2_avg[action_index] + clip_value_Q2 - y).square()),
          this->sdsac_temperature_alpha *(
              new_entropy.detach() - target_entropy).detach()};
}

void BaseSDSACAgent::update_parameters(
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
    for (const torch::OrderedDict<std::string, torch::Tensor>::Item &pair :
         this->critic->named_parameters(true)) {
      const std::string &name = pair.key();
      torch::Tensor param_main = pair.value();
      torch::Tensor param_avg =
          this->critic_Q1_avg->named_parameters(true)[name];
      param_avg.mul_(1.0 - this->sdsac_smoothing_tau);
      param_avg.add_(this->sdsac_smoothing_tau * param_main);
    }
    for (const torch::OrderedDict<std::string, torch::Tensor>::Item &pair :
         this->critic_Q2_main->named_parameters(true)) {
      const std::string &name = pair.key();
      torch::Tensor param_main = pair.value();
      torch::Tensor param_avg =
          this->critic_Q2_avg->named_parameters(true)[name];
      param_avg.mul_(1.0 - this->sdsac_smoothing_tau);
      param_avg.add_(this->sdsac_smoothing_tau * param_main);
    }
  }
  this->alpha_optimizer->zero_grad();
  temperature_alpha_loss.backward();
  this->alpha_optimizer->step();
}

void BaseSDSACAgent::save_model() const {
  std::lock_guard lock(*this->model_mutex);
  std::string name = this->agentName();
  if (name.empty()) {
    std::cerr << "No agent to save." << std::endl;
    return;
  }
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  fs::path critic_Q1_path = fs::path(AI_AGENTS_DIR) / (name + "-critic_Q1.pt");
  fs::path critic_Q2_path = fs::path(AI_AGENTS_DIR) / (name + "-critic_Q2.pt");
  fs::path alpha_path = fs::path(AI_AGENTS_DIR) / (name + "-alpha.pt");
  torch::save(this->actor, actor_path.string());
  torch::save(this->critic_Q1_avg, critic_Q1_path.string());
  torch::save(this->critic_Q2_avg, critic_Q2_path.string());
  torch::save(this->sdsac_temperature_alpha, alpha_path.string());
}

void BaseSDSACAgent::load_model() {
  std::lock_guard lock(*this->model_mutex);
  std::string name = this->agentName();
  if (name.empty()) {
    return;
  }
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  fs::path critic_Q1_path = fs::path(AI_AGENTS_DIR) / (name + "-critic_Q1.pt");
  fs::path critic_Q2_path = fs::path(AI_AGENTS_DIR) / (name + "-critic_Q2.pt");
  fs::path alpha_path = fs::path(AI_AGENTS_DIR) / (name + "-alpha.pt");
  if (!fs::exists(actor_path) || !fs::exists(critic_Q1_path) ||
      !fs::exists(critic_Q2_path) || !fs::exists(alpha_path)) {
    return;
  }
  try {
    torch::load(this->actor, actor_path.string(), this->device);
    torch::load(this->critic, critic_Q1_path.string(), this->device);
    torch::load(this->critic_Q2_main, critic_Q2_path.string(), this->device);
    torch::load(this->critic_Q1_avg, critic_Q1_path.string(), this->device);
    torch::load(this->critic_Q2_avg, critic_Q2_path.string(), this->device);
    torch::load(this->sdsac_temperature_alpha, alpha_path.string(),
                this->device);
  } catch (const std::exception &) {
    std::cerr << "Failed to load model for agent: " << name << std::endl;
    return;
  }
  std::cout << "Loaded model: " << name << std::endl;
}
} // namespace ai_pass_selector
