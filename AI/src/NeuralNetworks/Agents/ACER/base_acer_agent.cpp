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
    this->actor_avg->to(this->device);
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
  return this->actor->forward(x).dot(this->critic->forward(x));
}

torch::Tensor BaseACERAgent::get_value_avg(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(this->device).to(torch::kFloat32);
  return this->actor_avg->forward(x).dot(this->critic->forward(x));
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
  return action_probs.multinomial(1).squeeze(-1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
BaseACERAgent::compute_losses_and_accumulate_gradients(
    const torch::Tensor &reward,          // Shape []
    torch::Tensor Q_ret,                  // Shape []
    const torch::Tensor &policy_main,     // Shape [NR_PASSES]
    const torch::Tensor &policy_avg,      // Shape [NR_PASSES]
    const torch::Tensor &Q_values,        // Shape [NR_PASSES]
    const torch::Tensor &original_policy, // Shape [NR_PASSES]
    unsigned int action_index             // Shape []
) {
  std::vector<torch::Tensor> actor_parameters = this->actor->parameters(true);
  Q_ret = (reward + this->discount_factor * Q_ret).detach();
  torch::Tensor Vi = Q_values.dot(policy_main).detach();
  ////////////////////////////////////////////////////////////////////////////
  /// 1. Computing quantities needed for trust region updating
  torch::Tensor importance_weights =
      policy_main / (original_policy + DIVISION_BY_ZERO_BLOCK); // ρi(a)
  torch::Tensor truncated_importance_weights = torch::min(
      this->acer_truncation_threshold_c, importance_weights); // min{c,ρi(a)}
  torch::Tensor g_summand_top =
      truncated_importance_weights[action_index].detach() // min{c,ρi(ai)}
      * policy_main[action_index].log()                   // logf(ai|φθ(xi))
      * (Q_ret - Vi).detach();                            // (Qret − Vi)
  torch::Tensor g_summand_bottom =
      (1.0 - this->acer_truncation_threshold_c / importance_weights)
          .clamp_min(0.0)
          .detach()               // max(0,[1-c/ρi(a)])
      * policy_main.detach()      // f(a|φθ(xi))
      * policy_main.log()         // logf(a|φθ(xi))
      * (Q_values - Vi).detach(); // (Qθv(xi,a)−Vi)
  torch::Tensor g_scalar = g_summand_top + g_summand_bottom.sum();
  torch::Tensor k_scalar =
      this->compute_KL_divergence(policy_avg,
                                  policy_main); // DKL[f(·|φθa(xi))‖f(·|φθ(xi))]
  // Retain graph must be set to true so that the network of the actor can be
  // reused to compute k_vector.
  torch::autograd::variable_list g_gradients =
      torch::autograd::grad({g_scalar}, actor_parameters, {}, true);
  std::vector<torch::Tensor> g_flattened;
  for (const torch::Tensor &g_gradient : g_gradients) {
    g_flattened.push_back(g_gradient.contiguous().view(-1));
  }
  torch::Tensor g_vector = torch::cat(g_flattened);
  torch::autograd::variable_list k_gradients =
      torch::autograd::grad({k_scalar}, actor_parameters, {}, true);
  std::vector<torch::Tensor> k_flattened;
  for (const torch::Tensor &k_gradient : k_gradients) {
    k_flattened.push_back(k_gradient.contiguous().view(-1));
  }
  torch::Tensor k_vector = torch::cat(k_flattened);
  ////////////////////////////////////////////////////////////////////////////
  /// 2. Accumulating gradients with regard to θ and θv
  /// 3. Update Retrace target
  return {
      -g_vector + ((k_vector.dot(g_vector) - this->acer_trust_region_delta) /
                   (k_vector.square().sum() + DIVISION_BY_ZERO_BLOCK))
                          .clamp_min(0.0) *
                      k_vector,                  // g-max{0,(kTg−δ)/(‖k‖^2)}k
      (Q_ret - Q_values[action_index]).square(), // (Qret-Qθv(xi,a))^2
      Vi + truncated_importance_weights[action_index] *
               (Q_ret - Q_values[action_index]) // ρi(Qret−Qθv(xi,ai))+Vi
  };
}

void BaseACERAgent::update_parameters(const torch::Tensor &actor_gradients,
                                      const torch::Tensor &critic_loss) {
  this->actor_optimizer->zero_grad();
  std::vector<torch::Tensor> actor_parameters = this->actor->parameters(true);
  unsigned int offset = 0;
  for (unsigned int param_idx = 0; param_idx < actor_parameters.size();
       param_idx++) {
    torch::Tensor &actor_parameter_tensor = actor_parameters[param_idx];
    unsigned int nr_trainable_parameters = actor_parameter_tensor.numel();
    torch::Tensor gradient_slice =
        actor_gradients.slice(0, offset, offset + nr_trainable_parameters)
            .reshape(actor_parameter_tensor.sizes());
    if (actor_parameter_tensor.grad().defined()) {
      actor_parameter_tensor.mutable_grad() += gradient_slice;
    } else {
      actor_parameter_tensor.mutable_grad() = gradient_slice.clone();
    }
    offset += nr_trainable_parameters;
  }
  this->actor_optimizer->step();
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
  {
    torch::NoGradGuard no_grad_guard;
    for (const torch::OrderedDict<std::string, torch::Tensor>::Item &pair :
         this->actor->named_parameters(true)) {
      const std::string &name = pair.key();
      torch::Tensor param_main = pair.value();
      torch::Tensor param_avg = this->actor_avg->named_parameters(true)[name];
      param_avg.mul_(this->acer_soft_update_alpha);
      param_avg.add_((1.0 - this->acer_soft_update_alpha) * param_main);
    }
  }
}

unsigned int
BaseACERAgent::select_greedy_action(const torch::Tensor &observation) {
  return this->actor_avg
      ->forward(observation.to(this->device).to(torch::kFloat32))
      .argmax(/*dim=*/-1)
      .to(torch::kInt32)
      .detach()
      .item<int>();
}
void BaseACERAgent::save_model() const {
  std::lock_guard lock(*this->model_mutex);
  std::string name = this->agentName();
  if (name.empty()) {
    std::cerr << "No agent to save." << std::endl;
    return;
  }
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  fs::path critic_path = fs::path(AI_AGENTS_DIR) / (name + "-critic.pt");
  torch::save(this->actor_avg, actor_path.string());
  torch::save(this->critic, critic_path.string());
}

void BaseACERAgent::load_model() {
  std::lock_guard lock(*this->model_mutex);
  std::string name = this->agentName();
  if (name.empty()) {
    return;
  }
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  fs::path critic_path = fs::path(AI_AGENTS_DIR) / (name + "-critic.pt");
  if (!fs::exists(actor_path) || !fs::exists(critic_path)) {
    return;
  }
  try {
    torch::load(this->actor, actor_path.string(), this->device);
    torch::load(this->actor_avg, actor_path.string(), this->device);
    torch::load(this->critic, critic_path.string(), this->device);
  } catch (const std::exception &) {
    std::cerr << "Failed to load model for agent: " << name << std::endl;
    return;
  }
  std::cout << "Loaded model: " << name << std::endl;
}
} // namespace ai_pass_selector
