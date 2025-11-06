#include "NeuralNetworks/Agents/SAC/base_sac_agent.hpp"

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

bool BaseSACAgent::initialize(const torch::nn::Sequential &actor,
                              const torch::nn::Sequential &critic) {
  throw std::runtime_error("BaseSACAgent has 4 critics and must be initialized "
                           "using the 4-ciritc initialization method, not the "
                           "standard 1-critic initialization method.");
}
void BaseSACAgent::update_parameters(const torch::Tensor &actor_loss,
                                     const torch::Tensor &critic_loss) const {
  throw std::runtime_error(
      "BaseSACAgent has 4 critics and must be updated using the 4-critic "
      "update method, not the standard 1-critic update method.");
}
std::pair<torch::Tensor, torch::Tensor>
BaseSACAgent::forward(const torch::Tensor &observation) {
  throw std::runtime_error(
      "BaseSACAgent does not implement the standard Actor-Critic forward "
      "method with 2 return tensors. Use sac_forward instead.");
}
torch::Tensor BaseSACAgent::get_value(const torch::Tensor &observation) {
  throw std::runtime_error(
      "BaseSACAgent does not implement the standard Actor-Critic get_value "
      "method. Use sac_get_values instead.");
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseSACAgent::select_action(const torch::Tensor &observation) {
  throw std::runtime_error(
      "BaseSACAgent does not implement the standard Actor-Critic select_action "
      "method with 4 return tensors. Use sac_select_action instead.");
}

BaseSACAgent::BaseSACAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {
  this->sac_temperature_alpha =
      GLOBAL_PARAMS["sac_temperature_alpha"].to_double();
  this->sac_shared_learning_rate =
      GLOBAL_PARAMS["sac_shared_learning_rate"].to_double();
  this->sac_smoothing_tau = GLOBAL_PARAMS["sac_smoothing_tau"].to_double();
  if (this->sac_temperature_alpha <= 0.0) {
    this->sac_temperature_alpha = std::nullopt;
  }
}

bool BaseSACAgent::initialize(const torch::nn::Sequential &actor,
                              const torch::nn::Sequential &critic,
                              const torch::nn::Sequential &critic_Q2_main,
                              const torch::nn::Sequential &critic_Q1_avg,
                              const torch::nn::Sequential &critic_Q2_avg) {
  try {
    this->actor = actor;
    this->critic = critic;
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
    this->critic_V_avg->to(this->device);
    this->critic_Q1->to(this->device);
    this->critic_Q2->to(this->device);
    this->actor_optimizer = std::shared_ptr(
        std::move(makeOptimizer(this->actor_optimizer_type, this->actor,
                                this->sac_shared_learning_rate)));
    this->critic_optimizer = std::shared_ptr(
        std::move(makeOptimizer(this->critic_optimizer_type, this->critic,
                                this->sac_shared_learning_rate)));
    this->critic_Q1_optimizer = std::shared_ptr(
        std::move(makeOptimizer(this->critic_optimizer_type, this->critic_Q1,
                                this->sac_shared_learning_rate)));
    this->critic_Q2_optimizer = std::shared_ptr(
        std::move(makeOptimizer(this->critic_optimizer_type, this->critic_Q1,
                                this->sac_shared_learning_rate)));
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}
void BaseSACAgent::update_parameters(
    const torch::Tensor &actor_loss, const torch::Tensor &critic_loss,
    const torch::Tensor &critic_Q1_loss,
    const torch::Tensor &critic_Q2_loss) const {
  std::lock_guard lock(*this->model_mutex);
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
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
          this->critic_V_avg->named_parameters(/*recurse=*/true)[name];
      param_avg.mul_(1.0 - this->sac_smoothing_tau);
      param_avg.add_(this->sac_smoothing_tau * param_main);
    }
  }
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
BaseSACAgent::sac_forward(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(this->device).to(torch::kFloat32);
  return {this->actor->forward(x), this->critic->forward(x),
          this->critic_V_avg->forward(x)};
}
std::tuple<torch::Tensor, torch::Tensor>
BaseSACAgent::sac_Q_forward(const torch::Tensor &observation,
                            const torch::Tensor &action) {
  torch::Tensor x1 = observation.to(this->device).to(torch::kFloat32);
  torch::Tensor x2 = action.to(this->device).to(torch::kFloat32);
  return {this->critic_Q1->forward(x1, x2), this->critic_Q2->forward(x1, x2)};
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
BaseSACAgent::sac_select_action(const torch::Tensor &observation) {
  auto [action_probs, V_main, V_avg] = this->sac_forward(observation);
}

} // namespace ai_pass_selector