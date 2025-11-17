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
  this->acer_trust_region_delta =
      torch::tensor(GLOBAL_PARAMS["acer_trust_region_delta"].to_double(),
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
  if ((action_probs < 0).any().item<bool>()) {
    throw std::runtime_error("ACER action_probs contains x<0.");
  }
  if (torch::isinf(action_probs).any().item<bool>()) {
    throw std::runtime_error("ACER action_probs contains Inf.");
  }
  if (torch::isnan(action_probs).any().item<bool>()) {
    throw std::runtime_error("ACER action_probs contains NaN.");
  }
  return action_probs.multinomial(/*num_samples=*/1).squeeze(-1); // Shape []
}

void BaseACERAgent::compute_losses_and_accumulate_gradients(
    int k,                                             // Shape []
    const torch::Tensor &rewards,                      // Shape [k]
    torch::Tensor Q_ret,                               // Shape []
    const torch::Tensor &policies_main,                // Shape [k, NR_PASSES]
    const torch::Tensor &policies_avg,                 // Shape [k, NR_PASSES]
    const torch::Tensor &Q_values_list,                // Shape [k, NR_PASSES]
    const torch::Tensor &original_policies, // Shape [k, NR_PASSES]
    const std::vector<unsigned int> &action_indices    // Shape [k]
) {
  for (int i = k - 1; i >= 0; i--) {
    Q_ret = rewards[i] + this->discount_factor * Q_ret;
    torch::Tensor Vi = Q_values_list[i].dot(policies_main[i]);
    unsigned int action_index = action_indices[i];
    ////////////////////////////////////////////////////////////////////////////
    /// 1. Computing quantities needed for trust region updating
    torch::Tensor g_summand_top =
        torch::min(this->acer_truncation_threshold_c,
                   truncated_importance_weights[i][action_index])
            .detach()                          // min{c,ρi(ai)}
        * policies_main[i][action_index].log() // logf(ai|φθ(xi))
        * (Q_ret - Vi).detach();               // (Qret − Vi)
    torch::Tensor g_summand_bottom =
        (1.0 -
         this->acer_truncation_threshold_c / truncated_importance_weights[i])
            .clamp_min(0.0)
            .detach()                       // [1-c/ρi(ai)]+
        * policies_main[i].detach()         // f(a|φθ(xi))
        * policies_main[i].log()            // logf(a|φθ(xi))
        * (Q_values_list[i] - Vi).detach(); // (Qθv(xi,a)−Vi)
    torch::Tensor g_scalar = g_summand_top + g_summand_bottom.sum();
    // Technically the ACER trainer should detach the exponentially moving
    // average policy, but we are detaching it here again just to be sure.
    torch::Tensor k_scalar = this->compute_KL_divergence(
        /*policy_p=*/policies_avg[i].detach(),
        /*policy_q=*/policies_main[i]); // DKL[f(·|φθa(xi))‖f(·|φθ(xi))]
    // Turn g_scalar and k_scalar into g_vector and k_vector using
    // differentiation like in the ACER algorithm paper.
    std::vector<torch::Tensor> actor_parameters =
        this->actor->parameters(/*recurse=*/true);
    torch::autograd::variable_list g_gradients = torch::autograd::grad(
        /*outputs=*/{g_scalar}, /*inputs=*/actor_parameters,
        /*grad_outputs=*/{},
        /*retain_graph=*/true);
    std::vector<torch::Tensor> g_flattened;
    for (const torch::Tensor &g_gradient : g_gradients) {
      g_flattened.push_back(g_gradient.contiguous().view(-1));
    }
    torch::Tensor g_vector = torch::cat(g_flattened);
    auto k_gradients = torch::autograd::grad(
        /*outputs=*/{k_scalar}, /*inputs=*/actor_parameters,
        /*grad_outputs=*/{},
        /*retain_graph=*/true);
    std::vector<torch::Tensor> k_flattened;
    for (const torch::Tensor &k_gradient : k_gradients) {
      k_flattened.push_back(k_gradient.contiguous().view(-1));
    }
    torch::Tensor k_vector = torch::cat(k_flattened);
    ////////////////////////////////////////////////////////////////////////////
    /// 2. Accumulating gradients with regard to θ and θv
    torch::Tensor adjusted_actor_gradients =
        g_vector // g
        - std::max(0.0f,
                   ((k_vector.dot(g_vector) - this->acer_trust_region_delta) /
                    (k_vector.square().sum() + DIVISION_BY_ZERO_BLOCK))
                       .item<float>()) // max{0,(kTg−δ)/(‖k‖^2)}
              * k_vector;              // k
    // Assign adjusted gradients back to actor parameters
    unsigned int offset = 0;
    for (unsigned int param_idx = 0; param_idx < actor_parameters.size();
         param_idx++) {
      torch::Tensor &actor_parameter_tensor = actor_parameters[param_idx];
      unsigned int nr_trainable_parameters = actor_parameter_tensor.numel();
      actor_parameter_tensor.mutable_grad() +=
          adjusted_actor_gradients
              .slice(/*dim=*/0, /*start=*/offset,
                     /*end=*/offset + nr_trainable_parameters)
              .reshape(actor_parameter_tensor.sizes());
      offset += nr_trainable_parameters;
    }
    torch::Tensor critic_loss =
        (Q_ret - Q_values_list[i][action_index]).square();
    critic_loss.backward();
    ////////////////////////////////////////////////////////////////////////////
    /// 3. Update Retrace target
    Q_ret = Vi + truncated_importance_weights[i][action_index] *
                     (Q_ret - Q_values_list[i][action_index]);
  }
}

void BaseACERAgent::update_assuming_gradients_are_computed() {
  this->actor_optimizer->step();
  this->critic_optimizer->step(); //
  {
    torch::NoGradGuard no_grad_guard;
    for (const torch::OrderedDict<std::string, torch::Tensor>::Item &pair :
         this->actor->named_parameters(/*recurse=*/true)) {
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
