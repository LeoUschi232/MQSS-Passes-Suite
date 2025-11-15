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
      torch::tensor(GLOBAL_PARAMS["acer_truncation_threshold_c"].to_double(),
                    GLOBAL_TENSOR_OPTIONS);
  this->acer_soft_update_alpha =
      GLOBAL_PARAMS["acer_soft_update_alpha"].to_double();
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

void BaseACERAgent::reset_gradients() {
  std::lock_guard lock(*this->model_mutex);
  this->actor_optimizer->zero_grad();
  this->critic_optimizer->zero_grad();
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

void BaseACERAgent::compute_losses_and_accumulate_gradients(
    int k,                                             // Nr taken steps
    const torch::Tensor &rewards,                      // Shape [k]
    torch::Tensor Q_ret,                               // Shape []
    const torch::Tensor &policies_main,                // Shape [k, NR_PASSES]
    const torch::Tensor &policies_avg,                 // Shape [k, NR_PASSES]
    const torch::Tensor &Q_values_list,                // Shape [k, NR_PASSES]
    const torch::Tensor &truncated_importance_weights, // Shape [k]
    const std::vector<unsigned int> &action_indices    // Shape [k]
) {
  for (int i = k - 1; i >= 0; i--) {
    Q_ret = (rewards[i] + this->discount_factor * Q_ret).detach();
    torch::Tensor Vi = Q_values_list[i].dot(policies_main[i]).detach();
    unsigned int action_index = action_indices[i];
    ////////////////////////////////////////////////////////////////////////////
    /// Computing quantities needed for trust region updating
    torch::Tensor g_summand_top =
        torch::min(this->acer_truncation_threshold_c,
                   truncated_importance_weights[i][action_index])
            .detach()                          // min{c,ρi(ai)}
        * policies_main[i][action_index].log() // ∇φθ(xi)logf(ai|φθ(xi))
        * (Q_ret - Vi).detach();               // (Qret − Vi)
    torch::Tensor g_summand_bottom =
        (1.0 -
         this->acer_truncation_threshold_c / truncated_importance_weights[i])
            .clamp_min(0.0)
            .detach()               // [1-c/ρi(ai)]+
        * policies_main[i].detach() // f(a|φθ(xi))
        * policies_main[i].log()    // ∇φθ(xi)logf(a|φθ(xi))
        * (Q_values_list[i][action_index] - Vi).detach(); // (Qθv(xi,ai)−Vi)
    torch::Tensor quantity_g = g_summand_top + g_summand_bottom.sum();
    // Technically the ACER trainer should detach the exponentially moving
    // average policy, but we are detaching it here again just to be sure.
    torch::Tensor quantity_k =
        this->compute_KL_divergence(policies_avg[i].detach(), policies_main[i]);
    torch::Tensor actor_loss = quantity_g - torch::max(
      torch::tensor(0.0,  GLOBAL_TENSOR_OPTIONS), (quantity_k.dot(quantity_g) -this->acer_trust_region_delta)
      )

    ////////////////////////////////////////////////////////////////////////////
    actor_loss.backward();
    critic_loss.backward();
  }
}

void BaseACERAgent::update_assuming_gradients_are_computed() {
  this->actor_optimizer->step();
  this->critic_optimizer->step();
  {
    torch::NoGradGuard no_grad_guard;
    for (const auto &pair : this->actor->named_parameters(/*recurse=*/true)) {
      const std::string &name = pair.key();
      torch::Tensor param_main = pair.value();
      torch::Tensor param_avg =
          this->actor_avg->named_parameters(/*recurse=*/true)[name];
      param_avg.mul_(this->acer_soft_update_alpha);
      param_avg.add_((1.0 - this->acer_soft_update_alpha) * param_main);
    }
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
