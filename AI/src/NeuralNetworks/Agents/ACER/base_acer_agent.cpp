#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Utils includes
#include "Utils/info_utils.hpp"

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;
BaseACERAgent::BaseACERAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {
  this->acer_truncation_threshold_c =
      GLOBAL_PARAMS["acer_truncation_threshold_c"].to_double();
  this->acer_trust_region_delta =
      GLOBAL_PARAMS["acer_trust_region_delta"].to_double();
}

bool BaseACERAgent::initialize(
    const torch::nn::Sequential &actor_main,
    const torch::nn::Sequential &actor_avg,
    const torch::nn::Sequential &critic_Q_estimator) {
  try {
    this->actor = actor_main;
    this->actor_avg = actor_avg;
    this->critic = critic_Q_estimator;
    // Load the model before putting it to the device to avoid device
    // scheduling issus.
    this->load_model();
    this->register_module("actor", this->actor);
    this->register_module("actor_avg", this->actor_avg);
    this->register_module("critic", this->critic);
    this->critic->to(this->device);
    this->actor->to(this->device);
    this->actor_optimizer = std::shared_ptr(std::move(makeOptimizer(
        this->actor_optimizer_type, this->actor, this->actor_learning_rate)));
    this->critic_optimizer = std::shared_ptr(
        std::move(makeOptimizer(this->critic_optimizer_type, this->critic,
                                this->critic_learning_rate)));
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
BaseACERAgent::forward(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(this->device).to(torch::kFloat32);
  return {this->actor->forward(x), this->actor_avg->forward(x),
          this->critic->forward(x)};
}

torch::Tensor BaseACERAgent::get_value(const torch::Tensor &observation) {
  auto [policy_main, policy_avg, Q_values] = this->forward(
      /*observation=*/observation.to(this->device).to(torch::kFloat32));
  return policy_avg.detach().dot(Q_values.detach()).detach().unsqueeze(-1);
}

std::pair<torch::Tensor, torch::Tensor>
BaseACERAgent::select_action(const torch::Tensor &action_probs) {
  return {
      action_probs.multinomial(/*num_samples=*/1).squeeze(-1), // Shape []
      -(action_probs * action_probs.log())
           .sum(/*dim=*/-1)
           .squeeze(-1) // Shape []
  };
}

unsigned int
BaseACERAgent::select_greedy_action(const torch::Tensor &observation) {
  return this->actor_avg->forward(observation)
      .argmax(/*dim=*/-1)
      .to(torch::kInt32)
      .detach()
      .item<int>();
}

} // namespace ai_pass_selector
