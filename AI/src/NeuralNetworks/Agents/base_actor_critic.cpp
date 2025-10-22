#include "NeuralNetworks/Agents/base_actor_critic.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/info_utils.hpp"
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
  const torch::Tensor action_indexes_unsqueezed =
      action_probs.multinomial(/*num_samples=*/1);
  const torch::Tensor action_index = action_indexes_unsqueezed.squeeze(-1);

  // For advantage compute log π(a_t|s_t) for the sampled actions.
  // Gather extracts the values at specified indexes along the specified axis.
  // Parameter indexes must have the same nr of axes as the input tensor, here
  // each has 2 axes.
  // log_action_probs ~ [NR_PASSES]
  // X.gather(dim=-1, indexes=action_indexes_unsqueezed) ~ [1]
  // X.squeeze(dim=-1) ~ []
  const torch::Tensor log_action_probs = action_probs.log();
  const torch::Tensor squeezed_log_action_probs =
      log_action_probs.gather(/*dim=*/-1, /*indexes=*/action_indexes_unsqueezed)
          .squeeze(-1);

  // Entropy formula H = -sum_{x}(p(x)*log(p(x)))
  // action_probs * log_action_probs ~ [NR_PASSES]
  // -X.sum(dim=-1) ~ [1]
  const torch::Tensor entropy =
      -(action_probs * log_action_probs).sum(/*dim=*/-1);
  return {
      action_index,              // Shape []
      squeezed_log_action_probs, // Shape []
      state_value,               // Shape []
      entropy                    // Shape [1]
  };
}

std::pair<torch::Tensor, torch::Tensor>
BaseActorCritic::get_losses(const torch::Tensor &rewards,          // Shape [T]
                            const torch::Tensor &log_action_probs, // Shape [T]
                            const torch::Tensor &state_values, // Shape [T+1]
                            const torch::Tensor &entropy,      // Shape [T]
                            const double discount_factor,
                            const double gae_hyperparameter,
                            const double entropy_coefficient) {
  // Let T = final timestep of an episode.
  // An episode generates T+1 states from S_0 to S_T.
  // An episode generates T actions from A_1 to A_T.
  // An episode generates T rewards from R_1 to R_T.
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
    torch::Tensor delta_t =
        rewards[t] - state_values[t] + discount_factor * state_values[t + 1];

    // The generalized advantage estimation defined by Schulman et al is:
    // A_gae = sum_{l=0}^{\infty} (gamma * lamda)^l * delta_{t+l}
    A_gae = discount_factor * gae_hyperparameter * A_gae + delta_t;
    advantages[t] = A_gae;
  }

  // Give a bonus for higher entropy to encourage exploration.
  // The equation for the policy performance measure is:
  // J(θ) = (1/N) * sum_{t=0}^{N-1} (ln π_θ(a_t|s_t) * A(s_t, a_t))
  auto actor_loss = -(log_action_probs * advantages.detach()).mean() -
                    entropy_coefficient * entropy.mean();

  // The equation for the Value function performance measure is:
  // J(w) = (1/N) * sum_{t=0}^{N-1} (A(s_t, a_t)^2)
  auto critic_loss = advantages.pow(2).mean();
  return {actor_loss, critic_loss};
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
  fs::path critic_path = fs::path(AI_AGENTS_DIR) / (name + "-critic.pt");
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  torch::save(this->critic, critic_path.string());
  torch::save(this->actor, actor_path.string());
}

void BaseActorCritic::load_model() {
  std::lock_guard lock(*this->model_mutex);
  std::string name = this->agentName();
  if (name.empty()) {
    return;
  }
  fs::path critic_path = fs::path(AI_AGENTS_DIR) / (name + "-critic.pt");
  fs::path actor_path = fs::path(AI_AGENTS_DIR) / (name + "-actor.pt");
  if (!fs::exists(critic_path) || !fs::exists(actor_path)) {
    return;
  }
  torch::load(this->critic, critic_path.string(), this->device);
  torch::load(this->actor, actor_path.string(), this->device);
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

} // namespace ai_pass_selector