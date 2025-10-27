#include "NeuralNetworks/Agents/base_actor_critic.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/info_utils.hpp"
#include "Utils/passes_utils.hpp"
#include "Utils/tensor_utils.hpp"

// Standard library includes
#include <cmath>
#include <memory>
#include <tuple>
#include <utility>

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

BaseActorCritic::BaseActorCritic(unsigned int max_qubits)
    : AbstractAgent(max_qubits) {
  this->device = GLOBAL_PARAMS["device"].to_device_type();
  this->actor_learning_rate = GLOBAL_PARAMS["actor_learning_rate"].to_double();
  this->critic_learning_rate =
      GLOBAL_PARAMS["critic_learning_rate"].to_double();
  this->actor_optimizer_type =
      static_cast<OptimizerType>(GLOBAL_PARAMS["actor_optimizer_idx"].to_int());
  this->critic_optimizer_type = static_cast<OptimizerType>(
      GLOBAL_PARAMS["critic_optimizer_idx"].to_int());
  this->discount_factor = GLOBAL_PARAMS["discount_factor"].to_double();
  this->gae_hyperparameter = GLOBAL_PARAMS["gae_hyperparameter"].to_double();
  this->entropy_coefficient = GLOBAL_PARAMS["entropy_coefficient"].to_double();
}

bool BaseActorCritic::initialize(const torch::nn::Sequential &actor,
                                 const torch::nn::Sequential &critic) {
  try {
    this->actor = actor;
    this->critic = critic;
    // Load the model before putting it to the device to avoid device
    // scheduling issus.
    this->load_model();
    this->register_module("critic", this->critic);
    this->register_module("actor", this->actor);
    this->critic->to(this->device);
    this->actor->to(this->device);
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
BaseActorCritic::forward(const torch::Tensor &observation) {
  torch::Tensor x = observation.to(this->device).to(torch::kFloat32);
  // Do NOT reshape/flatten here.
  // Let the models handle shapes.
  return {this->actor->forward(x), this->critic->forward(x)};
}

torch::Tensor BaseActorCritic::get_value(const torch::Tensor &observation) {
  return this->critic->forward(
      observation.to(this->device).to(torch::kFloat32));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseActorCritic::select_action(const torch::Tensor &observation) {
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

unsigned int
BaseActorCritic::select_greedy_action(const torch::Tensor &observation) {
  auto [action_probs, _] = this->forward(observation);
  return action_probs.argmax(/*dim=*/-1).to(torch::kInt32).detach().item<int>();
}

torch::Tensor BaseActorCritic::compute_advantages(
    const torch::Tensor &rewards,     // Shape [T]
    const torch::Tensor &state_values // Shape [T+1]
) {
  // Let T = final timestep of an episode.
  // An episode generates T rewards from R_1 to R_T.
  // An episode generates T+1 states from S_0 to S_T.
  int T = rewards.size(0);
  const torch::TensorOptions options = rewards.options();
  torch::Tensor advantages = torch::zeros({T}, options);

  // Compute the advantages using Generalized Advantage Estimation.
  // Temporal Difference is a method used in Reinforcement Learning to estimate
  // the value function of a state based on the difference between the immediate
  // reward obtained from a current state and the estimated value of the next
  // state.
  torch::Tensor A_gae = torch::zeros({}, options);
  for (int t = T - 1; t >= 0; t--) {
    // Temporal Difference Error of V(s) with discount gamma is:
    // delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    // Barto & Sutton Reinforcement Learning page 121, equation (6.5)
    torch::Tensor delta_t = rewards[t] - state_values[t] +
                            this->discount_factor * state_values[t + 1];

    // The generalized advantage estimation defined by Schulman et al is:
    // A_gae = sum_{l=0}^{\infty} (gamma * lamda)^l * delta_{t+l}
    A_gae = this->discount_factor * this->gae_hyperparameter * A_gae + delta_t;
    advantages[t] = A_gae;
  }
  return advantages;
}
torch::Tensor
BaseActorCritic::compute_rewards_to_go(const torch::Tensor &rewards) {
  // Let T = final timestep of an episode.
  // An episode generates T rewards from R_1 to R_T.
  int T = rewards.size(0);
  if (T <= 0) {
    return torch::tensor({}, rewards.options());
  }
  const torch::TensorOptions options = rewards.options();
  torch::Tensor rewards_to_go = torch::zeros({T}, options);
  rewards_to_go[T - 1] = rewards[T - 1];
  for (int t = T - 2; t >= 0; t--) {
    rewards_to_go[t] =
        rewards[t] + this->discount_factor * rewards_to_go[t + 1];
  }
  return rewards_to_go;
}

void BaseActorCritic::update_parameters(
    const torch::Tensor &actor_loss, const torch::Tensor &critic_loss) const {
  std::lock_guard lock(*this->model_mutex);
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
}

void BaseActorCritic::save_model() const {
  std::lock_guard lock(*this->model_mutex);
  std::string name = this->agentName();
  if (name.empty()) {
    std::cerr << "No agent to save." << std::endl;
    return;
  }
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  fs::path critic_path = fs::path(AI_AGENTS_DIR) / (name + "-critic.pt");
  torch::save(this->actor, actor_path.string());
  torch::save(this->critic, critic_path.string());
}

void BaseActorCritic::load_model() {
  std::lock_guard lock(*this->model_mutex);
  std::string name = this->agentName();
  if (name.empty()) {
    return;
  }
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  fs::path critic_path = fs::path(AI_AGENTS_DIR) / (name + "-critic.pt");
  if (!fs::exists(critic_path) || !fs::exists(actor_path)) {
    return;
  }
  torch::load(this->actor, actor_path.string(), this->device);
  torch::load(this->critic, critic_path.string(), this->device);
  std::cout << "Loaded model: " << name << std::endl;
}

void BaseActorCritic::check_params(double tiny, double big) const {
  auto check = [&](const char *tag, const torch::nn::Sequential &network) {
    size_t total = 0, bad = 0;
    for (auto &keyvalue : network->named_parameters(/*recurse=*/true)) {
      const std::string &name = keyvalue.key();
      const torch::Tensor &value = keyvalue.value();
      total += value.numel();
      bool has_nan = value.isnan().any().item<bool>();
      bool has_inf = value.isinf().any().item<bool>();
      double abs_max = value.abs().max().item<double>();
      double abs_min_not_zero = 0.0;
      {
        torch::Tensor abs_value = value.abs();
        torch::Tensor flat = abs_value.view({-1});
        if (torch::Tensor non_zero = flat.index({flat.gt(0)});
            non_zero.numel() > 0) {
          abs_min_not_zero = non_zero.min().item<double>();
        }
      }
      if (has_nan || has_inf || abs_max > big ||
          (abs_min_not_zero > 0.0 && abs_min_not_zero < tiny)) {
        ++bad;
        std::cout << "[PARAM] " << tag << "." << name << " nan=" << has_nan
                  << " inf=" << has_inf << " abs_max=" << abs_max
                  << " abs_min_not_zero=" << abs_min_not_zero
                  << " shape=" << value.sizes() << "\n";
      }
    }
    std::cout << "[SUMMARY] " << tag << " total_params=" << total
              << " suspicious=" << bad << "\n";
  };
  if (this->actor) {
    check("actor", this->actor);
  }
  if (this->critic) {
    check("critic", this->critic);
  }
}

std::vector<std::function<std::unique_ptr<Pass>()>>
BaseActorCritic::select_passes_for_circuit(const fs::path &circuit_path) {
  QuantumCircuitEnvironment environment(this->max_qubits);
  if (!environment.register_quantum_circuit(circuit_path)) {
    std::cerr << "Failed to register quantum circuit: " << circuit_path
              << std::endl;
    return {};
  }
  std::vector<std::function<std::unique_ptr<Pass>()>> selected_passes;
  bool keep_going = true;
  while (keep_going) {
    unsigned int action_index = this->select_greedy_action(
        environment.get_observation_as_torch_tensor());
    auto [_, terminated, truncated] = environment.step(action_index);
    selected_passes.push_back(PASS_FUNCTIONS[action_index]);
    keep_going = !terminated && !truncated;
  }
  return selected_passes;
}

} // namespace ai_pass_selector