#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Utils includes
#include "Utils/info_utils.hpp"

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;
extern torch::TensorOptions GLOBAL_TENSOR_OPTIONS;

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

torch::Tensor BaseACERAgent::get_value_main(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(this->device).to(torch::kFloat32);
  return this->actor->forward(x).dot(this->critic->forward(x)).unsqueeze(-1);
}

torch::Tensor BaseACERAgent::get_value_avg(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(this->device).to(torch::kFloat32);
  return this->actor_avg->forward(x)
      .dot(this->critic->forward(x))
      .unsqueeze(-1);
}

torch::Tensor BaseACERAgent::select_action(const torch::Tensor &action_probs) {
  return action_probs.multinomial(/*num_samples=*/1).squeeze(-1); // Shape []
}

void BaseACERAgent::compute_losses(
    int k,                                            // Nr taken steps
    const torch::Tensor &rewards,                     // Shape [k]
    torch::Tensor Q_ret,                              // Shape []
    const torch::Tensor &policies_main,               // Shape [k, NR_PASSES]
    const torch::Tensor &policies_avg,                // Shape [k, NR_PASSES]
    const torch::Tensor &Q_values_list,               // Shape [k, NR_PASSES]
    const torch::Tensor &truncated_importance_weights // Shape [k]
) {
  torch::Tensor state_values = torch::zeros({k}, GLOBAL_TENSOR_OPTIONS);
  for (int i = k - 1; i >= 0; i--) {
    Q_ret = rewards[i] + this->discount_factor * Q_ret;
    state_values[i] = Q_values_list[i].dot(policies_main[i]);
  }
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
