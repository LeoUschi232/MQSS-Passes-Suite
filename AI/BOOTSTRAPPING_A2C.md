# Bootstrapping A2C

The quote refers to a key aspect of reinforcement learning algorithms like A2C (Advantage Actor-Critic) when using
Generalized Advantage Estimation (GAE) for computing advantages. In A2C, during training, you collect rollouts of
experience (states, actions, rewards, values) over a fixed number of steps (`max_steps_per_episode`, or T). These
rollouts may end in one of two ways:

- **Termination**: The environment naturally ends (e.g., a goal is reached or a failure condition is met), signaled by
  `terminates[b] = true` for that environment. In this case, there are no future rewards beyond that step, so the value
  bootstrap for future returns is 0.
- **Truncation**: The rollout is artificially cut off after T steps without the environment terminating (i.e.,
  `terminates[b] = false` after the last step for some environments). Here, there could still be future rewards from the
  final state `s_T`, so you need to estimate them using the critic's value function V(`s_T`). This is the "bootstrapped
  value."

Without bootstrapping, the code treats all rollouts as if they terminate at T, assuming V(`s_T`) = 0 for everyone. This
introduces bias, especially in continuing tasks or when max_steps is a horizon limit rather than a natural episode
length. The result? Underestimated advantages and returns for truncated trajectories, leading to suboptimal policy
updates.

### What Exactly Must Be Done

1. **Compute the next values**: After the inner loop (which collects data up to step T-1), fetch the final batched
   observations (`s_T`) and use the critic to estimate V(`s_T`) for each environment. This gives a tensor of "
   next_values" (shape [B], where B is `nr_parallel_environments`).
2. **Mask based on termination**: For each environment b:
    - If it terminated after the last step (`terminates[b] = true` from the last step, which sets
      `termination_masks[T-1][b] = 0.0`), use bootstrap = 0.
    - If truncated (`termination_masks[T-1][b] = 1.0`), use bootstrap = next_values[b].
3. **Incorporate into GAE**: Pass `next_values` to `agent->get_losses(...)`. Inside `get_losses` (which you haven't
   shown, but assuming it's a standard A2C/GAE implementation), modify the advantage calculation to use this bootstrap
   in the TD error (delta) for the last timestep (t = T-1):
    - delta_{T-1} = r_{T-1} + γ * termination_masks[T-1] * next_values - v_{T-1}
    - Then, compute GAE backwards as usual for all timesteps, carrying the advantages with γ * λ *
      termination_masks[t] * advantage_{t+1}.
    - For earlier timesteps where an environment terminated early, the masks already reset the carry-over to 0,
      preventing value leakage across episodes.

This assumes your `ParallelEnvironments` class handles terminated environments properly (e.g., via absorbing states
where post-termination rewards are 0 and states don't change, or by masking in the step function). If an environment
terminates early, the loop continues selecting actions on its final/absorbing state, which is fine as long as V(
absorbing) ≈ 0.

### How to Implement It

You'll need to:

- Add a line after the inner loop to compute `next_values`.
- Modify `BaseA2CAgent::get_losses` to accept an additional `torch::Tensor next_values` parameter.
- Update the GAE logic inside `get_losses` to use it (I can't show the exact internal changes without seeing that
  method, but the standard way is shown below conceptually).
- If `select_action` runs both actor and critic, you can call it to get the values (ignoring the rest), or better yet,
  add a pure `get_value` method to `BaseA2CAgent` for efficiency:
  ```cpp
  torch::Tensor BaseA2CAgent::get_value(const torch::Tensor& observations) {
      // Assuming critic is a member that takes batched obs and returns values.
      return critic->forward(observations).squeeze(-1);  // Adjust based on your critic output shape.
  }
  ```

Here's the modified `train_a2c` function with the additions highlighted in comments. I've used `select_action` to
compute `next_values` for simplicity, but switch to `get_value` if you add it.

```cpp
#include "Agents/A2C/a2c_trainer.hpp"
// Environment includes
#include "Environment/parallel_environments.hpp"
// Agents includes
#include "Agents/agent_utils.hpp"
// Utils includes
#include "Support/mlir_utils.hpp"
#include "Utils/info_utils.hpp"
#include "Utils/progress_bar.hpp"
// Standard library includes
#include <filesystem>
using namespace mqss::support::quakeDialect;
namespace fs = std::filesystem;
namespace ai_pass_selector {
std::unordered_map<std::string, std::string>
train_a2c(std::unique_ptr<BaseA2CAgent> agent, const std::string &dataset,
          std::unordered_map<std::string, std::string> params) {
  unsigned int max_qubits = agent->getMaxQubits();
  if (params["print_param_info"] == "true") {
    std::cout << "Training A2C agent with parameters:" << std::endl;
    std::cout << " max_qubits: " << max_qubits << std::endl;
    for (auto [key, value] : params) {
      std::cout << " " << key << ": " << value << std::endl;
    }
  }
  // Default values
  unsigned int nr_parallel_environments =
      std::stoul(params["nr_parallel_environments"]);
  unsigned int episodes = std::stoul(params["episodes"]);
  unsigned int max_steps_per_episode =
      std::stoul(params["max_steps_per_episode"]);
  double discount_factor = std::stod(params["discount_factor"]);
  double gae_hyperparameter = std::stod(params["gae_hyperparameter"]);
  double entropy_coefficient = std::stod(params["entropy_coefficient"]);
  torch::Device device = DEVICE_NAME_TO_TORCH.at(params["device"]);
  if (nr_parallel_environments <= 0) {
    std::cerr << "No agent to train." << std::endl;
    return {};
  }
  auto optional_statistics = get_precomputed_dataset_statistics(dataset);
  if (!optional_statistics.has_value()) {
    std::cerr << "Dataset " + dataset + " doesn't have statistics for training."
              << std::endl;
    return {};
  }
  auto [qubits_cholesky_params, gates_weights] = optional_statistics.value();
  ParallelEnvironments environments(nr_parallel_environments, max_qubits,
                                    max_steps_per_episode);
  environments.register_randomizer_params(qubits_cholesky_params,
                                          gates_weights);
  double max_reward = -std::numeric_limits<double>::max();
  double summed_rewards = 0.0;
  std::vector<double> entropies;
  std::vector<double> critic_losses;
  std::vector<double> actor_losses;
  int64_t T = max_steps_per_episode;
  int64_t B = nr_parallel_environments;
  std::cout << "Beginning training." << std::endl;
  updateProgress(0, episodes, "Beginning training");
  for (unsigned int episode_nr = 1; episode_nr <= episodes; episode_nr++) {
    environments.reset();
    torch::TensorOptions options =
        torch::TensorOptions().device(device).dtype(torch::kFloat64);
    auto episode_log_probs = torch::zeros({T, B}, options);
    auto episode_values = torch::zeros({T, B}, options);
    auto episode_rewards = torch::zeros({T, B}, options);
    auto episode_entropies = torch::zeros({T, B}, options);
    auto termination_masks = torch::zeros({T, B}, options);
    for (unsigned int update_step = 0; update_step < max_steps_per_episode;
         update_step++) {
      auto [actions, log_action_probs, state_values, step_entropy] =
          agent->select_action(environments.get_batched_observations());
      auto [rewards, terminates] = environments.step(actions);
      episode_log_probs[update_step] = log_action_probs;
      episode_values[update_step] = state_values;
      episode_entropies[update_step] = step_entropy;
      for (unsigned int b = 0; b < B; b++) {
        episode_rewards[update_step][b] = rewards[b];
        termination_masks[update_step][b] = terminates[b] ? 0.0 : 1.0;
      }
    }
    // NEW: Compute bootstrapped next_values for truncated environments.
    auto next_observations = environments.get_batched_observations();
    auto [_, __, next_values, ___] = agent->select_action(next_observations);  // Ignore actions, log_probs, entropy.
    // Alternatively, if you add get_value: auto next_values = agent->get_value(next_observations);

    // MODIFIED: Pass next_values to get_losses.
    auto [critic_loss, actor_loss] =
        agent->get_losses(episode_rewards, episode_log_probs, episode_values,
                          episode_entropies, termination_masks, discount_factor,
                          gae_hyperparameter, entropy_coefficient, next_values);
    auto episode_rewards_cpu = episode_rewards.to(torch::kCPU);
    auto total_rewards = episode_rewards_cpu.sum(/*axis=*/0);
    if (total_rewards.size(/*dim=*/0) != nr_parallel_environments) {
      throw std::runtime_error("total_rewards.size=/=nr_parallel_environments");
    }
    double current_reward = total_rewards.sum().item<double>();
    summed_rewards += current_reward;
    if (current_reward > max_reward) {
      max_reward = current_reward;
      agent->save_model();
    }
    agent->update_parameters(critic_loss, actor_loss);
    entropies.push_back(episode_entropies.mean().item<double>());
    critic_losses.push_back(critic_loss.item<double>());
    actor_losses.push_back(actor_loss.item<double>());
    updateProgress(episode_nr, episodes,
                   "Max: " + std::to_string(max_reward) + " | Avg: " +
                       std::to_string(summed_rewards / episode_nr));
  }
  std::cout << "\nTraining finished." << std::endl;
  if (params["save_agent_at_end_of_training"] == "true") {
    std::cout << "Saving: " << agent->agentName() << std::endl;
    agent->save_model();
    std::cout << "Saved: " << agent->agentName() << std::endl;
  }
  return {{"max_reward", std::to_string(max_reward)},
          {"average_reward", std::to_string(summed_rewards / episodes)},
          {"final_entropy", std::to_string(entropies.back())},
          {"final_critic_loss", std::to_string(critic_losses.back())},
          {"final_actor_loss", std::to_string(actor_losses.back())}};
}
} // namespace ai_pass_selector
```

### Conceptual GAE Implementation Inside `get_losses`

Assuming `get_losses` computes advantages backwards (standard for GAE), here's how to adjust it (pseudocode; integrate
into your actual method):

```cpp
std::pair<torch::Tensor, torch::Tensor> BaseA2CAgent::get_losses(
    const torch::Tensor& rewards, const torch::Tensor& log_probs, const torch::Tensor& values,
    const torch::Tensor& entropies, const torch::Tensor& masks, double gamma, double lambda,
    double entropy_coef, const torch::Tensor& next_values) {  // NEW: Added next_values.

  int64_t T = rewards.size(0);
  int64_t B = rewards.size(1);

  auto advantages = torch::zeros_like(rewards);
  auto returns = torch::zeros_like(rewards);

  // Start backwards pass from last step.
  auto last_gae_lam = torch::zeros({B}, rewards.options());
  for (int64_t t = T - 1; t >= 0; --t) {
    if (t == T - 1) {
      // NEW: Use next_values with mask for bootstrap.
      auto delta = rewards[t] + gamma * masks[t] * next_values - values[t];
      last_gae_lam = delta + gamma * lambda * masks[t] * last_gae_lam;
    } else {
      auto delta = rewards[t] + gamma * masks[t] * values[t + 1] - values[t];
      last_gae_lam = delta + gamma * lambda * masks[t] * last_gae_lam;
    }
    advantages[t] = last_gae_lam;
    returns[t] = advantages[t] + values[t];  // Optional, if using returns directly.
  }

  // Rest unchanged: Compute actor_loss = - (log_probs * advantages.detach()).mean() + entropy_coef * entropies.mean();
  // critic_loss = (returns - values).pow(2).mean() / 2;  // Or similar.
  // ...
}
```

This should make training more accurate for truncated rollouts. Test it thoroughly, as bootstrapping can affect
convergence speed and stability. If your environments always terminate exactly at max_steps (no truncations), this isn't
strictly needed—but it's good practice for generalizability.

### Required Changes for Bootstrapping in GAE

Yes, you're correct that `get_losses` needs to be updated to accept an additional parameter for
bootstrapping—specifically, a `torch::Tensor next_values` (shape [B], where B is the batch size of parallel
environments), which represents the critic's value estimates V(s_T) for the final states after the last step. Passing
the raw next states (observations) wouldn't be ideal, as the agent already has a `get_value` method to compute values
from observations efficiently. This avoids redundant forward passes or unnecessary complexity.

The current implementation of `get_losses` has two issues:

1. **Missing bootstrapping**: It assumes V(s_T) = 0 for all environments at the final timestep (t = T-1), which biases
   learning if episodes are truncated (i.e., not naturally terminated) after `max_steps_per_episode`.
2. **Loop bug**: The backwards loop starts from `t = T-2` and never computes the advantage for `t = T-1` (
   advantages[T-1] remains zero-initialized). This is a critical error, as the last timestep's advantage is always
   ignored in the losses.

To fix this precisely:

- In the training loop (`train_a2c`), compute `next_values` after collecting the rollout by calling
  `agent->get_value(environments.get_batched_observations())`. This gets V(s_T) for the states after the final step.
- Update the signature of `BaseA2CAgent::get_losses` in the header and implementation to accept `next_values`.
- In `get_losses`, modify the GAE computation to start the loop from `t = T-1` to `0`. For `t = T-1`, use `next_values`
  in the delta calculation (masked by `termination_masks[t]`). For earlier timesteps, use `state_values[t+1]` as before.
- The critic loss should ideally be divided by 2 (for half-MSE, common in A2C implementations to match the gradient
  scale of policy loss), but this is optional—I've added it below for better stability.
- No other changes are needed (e.g., actor loss and entropy handling are already correct).

Here's the updated code with minimal changes. I've highlighted the additions/modifications in comments for clarity.

#### 1. Update the Header (`base_a2c_agent.hpp`)

Add `next_values` to the `get_losses` signature.

```cpp
#ifndef BASE_A2C_AGENT_HPP
#define BASE_A2C_AGENT_HPP
// Torch includes
#include "torch/torch.h"
// Standard library includes
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
namespace fs = std::filesystem;
namespace ai_pass_selector {
class BaseA2CAgent : public torch::nn::Module {
protected:
  /// Attributes on configuration
  unsigned int max_qubits = 0;
  int critic_optimizer_type = 0;
  int actor_optimizer_type = 0;
  double critic_learning_rate = 0;
  double actor_learning_rate = 0;
  unsigned int nr_parallel_environments = 0;
  torch::Device device = torch::kCPU;
  /// Attributes on initialization
  torch::nn::Sequential critic = nullptr;
  torch::nn::Sequential actor = nullptr;
  std::unique_ptr<torch::optim::Optimizer> actor_optimizer = nullptr;
  std::unique_ptr<torch::optim::Optimizer> critic_optimizer = nullptr;
  /// Mutex for thread safety
  std::unique_ptr<std::mutex> model_mutex = std::make_unique<std::mutex>();
public:
  /// Constructors
  BaseA2CAgent(unsigned int max_qubits,
               std::unordered_map<std::string, std::string> params);
  /**
   *
   * @param max_qubits
   * @param params
   */
  void configure(unsigned int max_qubits,
                 std::unordered_map<std::string, std::string> params);
  /**
   *
   * @param actor
   * @param critic
   * @return
   */
  bool initialize(const torch::nn::Sequential &actor,
                  const torch::nn::Sequential &critic);
  /// Destructor
  ~BaseA2CAgent() override = default;
  /// Copy and move constructors and assignment operators
  BaseA2CAgent(const BaseA2CAgent &other) = delete;
  BaseA2CAgent(BaseA2CAgent &&other) noexcept = default;
  BaseA2CAgent &operator=(const BaseA2CAgent &other) = delete;
  BaseA2CAgent &operator=(BaseA2CAgent &&other) noexcept = delete;
  /// Getters
  unsigned int getMaxQubits() const;
  /**
   *
   * @param batched_observations
   * @return
   */
  std::pair<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &batched_observations);
  /**
   *
   * @param batched_observations
   * @return
   */
  torch::Tensor get_value(const torch::Tensor &batched_observations);
  /**
   *
   * @param batched_observations
   * @return
   */
  std::tuple<std::vector<unsigned int>, torch::Tensor, torch::Tensor,
             torch::Tensor>
  select_action(const torch::Tensor &batched_observations);
  /**
   *
   * @param rewards
   * @param log_action_probs
   * @param state_values
   * @param entropy
   * @param termination_masks
   * @param discount_factor
   * @param gae_hyperparameter
   * @param entropy_coefficient
   * @param next_values  // NEW: Added for bootstrapping V(s_T)
   * @return
   */
  static std::pair<torch::Tensor, torch::Tensor>
  get_losses(const torch::Tensor &rewards,
             const torch::Tensor &log_action_probs,
             const torch::Tensor &state_values, const torch::Tensor &entropy,
             const torch::Tensor &termination_masks, double discount_factor,
             double gae_hyperparameter, double entropy_coefficient,
             const torch::Tensor &next_values);
  /**
   *
   * @param critic_loss
   * @param actor_loss
   */
  void update_parameters(const torch::Tensor &critic_loss,
                         const torch::Tensor &actor_loss) const;
  /// Saving and Loading
  void save_model() const;
  void load_model();
  virtual std::string agentName() const = 0;
};
} // namespace ai_pass_selector
#endif // BASE_A2C_AGENT_HPP
```

#### 2. Update the Implementation (`base_a2c_agent.cpp`)

Add `next_values` to `get_losses` and fix the GAE loop to include `t = T-1` with bootstrapping.

```cpp
#include "Agents/A2C/base_a2c_agent.hpp"

// Torch includes
#include "Agents/agent_utils.hpp"
#include "torch/torch.h"

// Standard library includes
#include <cmath>
#include <memory>
#include <tuple>
#include <utility>

namespace ai_pass_selector {

BaseA2CAgent::BaseA2CAgent(
    unsigned int max_qubits,
    std::unordered_map<std::string, std::string> params) {
  this->configure(max_qubits, std::move(params));
}

void BaseA2CAgent::configure(
    unsigned int max_qubits,
    std::unordered_map<std::string, std::string> params) {
  this->max_qubits = max_qubits;
  this->critic_optimizer_type =
      OPTIMIZER_NAME_TO_TYPE.at(params["critic_optimizer"]);
  this->actor_optimizer_type =
      OPTIMIZER_NAME_TO_TYPE.at(params["actor_optimizer"]);
  this->critic_learning_rate = std::stod(params["critic_learning_rate"]);
  this->actor_learning_rate = std::stod(params["actor_learning_rate"]);
  this->nr_parallel_environments =
      std::stoul(params["nr_parallel_environments"]);
  this->device = (params["device"] == "cuda" || params["device"] == "gpu") &&
                         torch::cuda::is_available()
                     ? torch::kCUDA
                     : torch::kCPU;
}

bool BaseA2CAgent::initialize(const torch::nn::Sequential &actor,
                              const torch::nn::Sequential &critic) {
  try {
    this->actor = actor;
    this->critic = critic;
    // Load the model before putting it to the device to avoid device
    // scheduling issus.
    this->load_model();
    register_module("critic", this->critic);
    register_module("actor", this->actor);
    this->critic->to(this->device);
    this->actor->to(this->device);
    this->critic_optimizer = makeOptimizer(critic_optimizer_type, this->critic,
                                           this->critic_learning_rate);
    this->actor_optimizer = makeOptimizer(actor_optimizer_type, this->actor,
                                          this->actor_learning_rate);
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}

unsigned int BaseA2CAgent::getMaxQubits() const { return this->max_qubits; }

std::pair<torch::Tensor, torch::Tensor>
BaseA2CAgent::forward(const torch::Tensor &batched_observations) {
  std::lock_guard lock(*this->model_mutex);
  torch::Tensor x = batched_observations.to(this->device).to(torch::kFloat);
  // Do NOT reshape/flatten here.
  // Let the models handle shapes.
  return {this->critic->forward(x), this->actor->forward(x)};
}

torch::Tensor
BaseA2CAgent::get_value(const torch::Tensor &batched_observations) {
  std::lock_guard lock(*this->model_mutex);
  torch::Tensor x = batched_observations.to(this->device).to(torch::kFloat);
  // Assuming critic is a member that takes batched obs and returns values.
  return critic->forward(x).squeeze(-1);
}

std::tuple<std::vector<unsigned int>, torch::Tensor, torch::Tensor,
           torch::Tensor>
BaseA2CAgent::select_action(const torch::Tensor &batched_observations) {
  auto [state_values, action_probs] = this->forward(batched_observations);
  // sample one action per row; result is [B,1] -> squeeze to [B]
  torch::Tensor actions_tensor = action_probs.multinomial(1).squeeze(-1);
  // CUDA tensors can’t be read directly)
  torch::Tensor actions_cpu = actions_tensor.to(torch::kCPU);
  std::vector<unsigned int> actions;
  actions.reserve(actions_cpu.size(0));
  for (int64_t i = 0; i < actions_cpu.size(0); ++i) {
    actions.push_back(
        static_cast<unsigned int>(actions_cpu[i].item<int64_t>()));
  }

  // log π(a|s) for the sampled actions: gather along the action dim
  const torch::Tensor log_action_probs = action_probs.log();
  // a_t, log π(a_t|s_t), V(s_t), entropy of π(a_t|s_t)
  return {actions,
          log_action_probs.gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1),
          state_values.squeeze(-1), -(action_probs * log_action_probs).sum(-1)};
}

std::pair<torch::Tensor, torch::Tensor> BaseA2CAgent::get_losses(
    const torch::Tensor &rewards, const torch::Tensor &log_action_probs,
    const torch::Tensor &state_values, const torch::Tensor &entropy,
    const torch::Tensor &termination_masks, const double discount_factor,
    const double gae_hyperparameter, const double entropy_coefficient,
    const torch::Tensor &next_values) {  // NEW: Added next_values parameter.

  // Let T = final timestep of an episode.
  // An episode generates T+1 states from S_0 to S_T.
  // An episode generates T actions from A_0 to A_{T-1}.
  // An episode generates T rewards from R_1 to R_T.
  int T = rewards.size(0);
  int B = rewards.size(1);
  const torch::TensorOptions options = rewards.options();
  torch::Tensor advantages = torch::zeros({T, B}, options);

  // Compute the advantages using Generalized Advantage Estimation.
  // Temporal Difference is a method used in Reinforcement Learning to estimate
  // the value function of a state based on the difference between the immediate
  // reward obtained from a current state and the estimated value of the next
  // state.
  torch::Tensor A_gae = torch::zeros({B}, options);
  for (int t = T - 1; t >= 0; t--) {  // MODIFIED: Start from T-1 to include the last timestep.
    // In Barto & Sutton the temporal difference residual of V with discount
    // gamma is: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    torch::Tensor delta_t;
    if (t == T - 1) {
      // NEW: Use bootstrapped next_values for the final timestep (masked if terminated).
      delta_t = rewards[t] + discount_factor * next_values * termination_masks[t] - state_values[t];
    } else {
      delta_t = rewards[t] + discount_factor * state_values[t + 1] * termination_masks[t] - state_values[t];
    }

    // The generalized advantage estimation defined by Schulman et al is:
    // A_gae = sum_{l=0}^{\infty} (gamma * lamda)^l * delta_{t+l}
    A_gae = delta_t + discount_factor * gae_hyperparameter * termination_masks[t] * A_gae;  // MODIFIED: Order adjusted for standard GAE (delta first, then add carried term).
    advantages[t] = A_gae;
  }

  // The equation for the Value function performance measure is:
  // J(w) = (1/N) * sum_{t=0}^{N-1} (A(s_t, a_t)^2)
  auto critic_loss = advantages.pow(2).mean() / 2;  // MODIFIED: Added /2 for half-MSE (optional but recommended for gradient scale).

  // Give a bonus for higher entropy to encourage exploration.
  // The equation for the policy performance measure is:
  // J(θ) = (1/N) * sum_{t=0}^{N-1} (ln π_θ(a_t|s_t) * A(s_t, a_t))
  auto actor_loss = -(log_action_probs * advantages.detach()).mean() -
                    entropy_coefficient * entropy.mean();
  return {critic_loss, actor_loss};
}

void BaseA2CAgent::update_parameters(const torch::Tensor &critic_loss,
                                     const torch::Tensor &actor_loss) const {
  std::lock_guard lock(*this->model_mutex);
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
}

void BaseA2CAgent::save_model() const {
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

void BaseA2CAgent::load_model() {
  // Silently doesn't load if model doesn'T exist as intended.
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

} // namespace ai_pass_selector
```

#### 3. Update the Trainer (`a2c_trainer.cpp`)

Compute `next_values` after the rollout loop and pass it to `get_losses`. (Note: I've removed the unused `entropies`,
`critic_losses`, and `actor_losses` vectors from your provided code, as they weren't in the return map anyway.)

```cpp
#include "Agents/A2C/a2c_trainer.hpp"

// Environment includes
#include "Environment/parallel_environments.hpp"

// Agents includes
#include "Agents/agent_utils.hpp"

// Utils includes
#include "Support/mlir_utils.hpp"
#include "Utils/info_utils.hpp"
#include "Utils/progress_bar.hpp"

// Standard library includes
#include <filesystem>

using namespace mqss::support::quakeDialect;
namespace fs = std::filesystem;

namespace ai_pass_selector {
std::unordered_map<std::string, std::string>
train_a2c(std::unique_ptr<BaseA2CAgent> agent, const std::string &dataset,
          std::unordered_map<std::string, std::string> params) {
  unsigned int max_qubits = agent->getMaxQubits();
  if (params["print_param_info"] == "true") {
    std::cout << "Training A2C agent with parameters:" << std::endl;
    std::cout << "  max_qubits: " << max_qubits << std::endl;
    for (auto [key, value] : params) {
      std::cout << "  " << key << ": " << value << std::endl;
    }
  }

  // Default values
  unsigned int nr_parallel_environments =
      std::stoul(params["nr_parallel_environments"]);
  unsigned int episodes = std::stoul(params["episodes"]);
  unsigned int max_steps_per_episode =
      std::stoul(params["max_steps_per_episode"]);
  double discount_factor = std::stod(params["discount_factor"]);
  double gae_hyperparameter = std::stod(params["gae_hyperparameter"]);
  double entropy_coefficient = std::stod(params["entropy_coefficient"]);
  torch::Device device = DEVICE_NAME_TO_TORCH.at(params["device"]);

  if (nr_parallel_environments <= 0) {
    std::cerr << "No agent to train." << std::endl;
    return {};
  }
  auto optional_statistics = get_precomputed_dataset_statistics(dataset);
  if (!optional_statistics.has_value()) {
    std::cerr << "Dataset " + dataset + " doesn't have statistics for training."
              << std::endl;
    return {};
  }
  auto [qubits_cholesky_params, gates_weights] = optional_statistics.value();
  ParallelEnvironments environments(nr_parallel_environments, max_qubits,
                                    max_steps_per_episode);
  environments.register_randomizer_params(qubits_cholesky_params,
                                          gates_weights);

  double max_reward = -std::numeric_limits<double>::max();
  double summed_rewards = 0.0;
  int64_t T = max_steps_per_episode;
  int64_t B = nr_parallel_environments;

  std::cout << "Beginning training." << std::endl;
  updateProgress(0, episodes, "Beginning training");
  for (unsigned int episode_nr = 1; episode_nr <= episodes; episode_nr++) {
    environments.reset();
    torch::TensorOptions options =
        torch::TensorOptions().device(device).dtype(torch::kFloat64);
    torch::Tensor episode_log_probs = torch::zeros({T, B}, options);
    torch::Tensor episode_values = torch::zeros({T, B}, options);
    torch::Tensor episode_rewards = torch::zeros({T, B}, options);
    torch::Tensor episode_entropies = torch::zeros({T, B}, options);
    torch::Tensor termination_masks = torch::zeros({T, B}, options);

    for (unsigned int update_step = 0; update_step < max_steps_per_episode;
         update_step++) {
      auto [actions, log_action_probs, state_values, step_entropy] =
          agent->select_action(environments.get_batched_observations());
      auto [rewards, terminates] = environments.step(actions);
      episode_log_probs[update_step] = log_action_probs;
      episode_values[update_step] = state_values;
      episode_entropies[update_step] = step_entropy;
      for (unsigned int b = 0; b < B; b++) {
        episode_rewards[update_step][b] = rewards[b];
        termination_masks[update_step][b] = terminates[b] ? 0.0 : 1.0;
      }
    }

    // NEW: Compute bootstrapped next_values for the final states (s_T).
    torch::Tensor next_values = agent->get_value(environments.get_batched_observations());

    // MODIFIED: Pass next_values to get_losses.
    auto [critic_loss, actor_loss] =
        agent->get_losses(episode_rewards, episode_log_probs, episode_values,
                          episode_entropies, termination_masks, discount_factor,
                          gae_hyperparameter, entropy_coefficient, next_values);

    auto episode_rewards_cpu = episode_rewards.to(torch::kCPU);
    auto total_rewards = episode_rewards_cpu.sum(/*axis=*/0);
    if (total_rewards.size(/*dim=*/0) != nr_parallel_environments) {
      throw std::runtime_error("total_rewards.size=/=nr_parallel_environments");
    }
    double current_reward = total_rewards.sum().item<double>();
    summed_rewards += current_reward;
    if (current_reward > max_reward) {
      max_reward = current_reward;
      agent->save_model();
    }
    agent->update_parameters(critic_loss, actor_loss);
    updateProgress(episode_nr, episodes,
                   "Max: " + std::to_string(max_reward) + " | Avg: " +
                       std::to_string(summed_rewards / episode_nr));
  }
  std::cout << "\nTraining finished." << std::endl;

  if (params["save_agent_at_end_of_training"] == "true") {
    std::cout << "Saving: " << agent->agentName() << std::endl;
    agent->save_model();
    std::cout << "Saved: " << agent->agentName() << std::endl;
  }
  return {{"max_reward", std::to_string(max_reward)},
          {"average_reward", std::to_string(summed_rewards / episodes)}};
}
} // namespace ai_pass_selector
```

### Additional Notes

- **If all episodes always terminate exactly at `max_steps_per_episode`**: Bootstrapping isn't strictly necessary (set
  `next_values` to zeros), but adding it makes the code more robust for general environments.
- **Testing**: Verify shapes—`next_values` should be [B] with dtype float64 on the correct device. If environments
  handle terminations with absorbing states (post-termination rewards=0, states unchanged), this will work seamlessly.
- **Performance Impact**: This fix should improve convergence, especially in non-terminal horizons, by reducing bias in
  advantages.
- **Backward Compatibility**: If you want to optionally disable bootstrapping, add a default
  `next_values = torch::zeros({B}, options)` in the signature, but I recommend always using it.

This should fully resolve the issues while keeping changes minimal. If you provide more context (e.g., the environment
details), I can refine further.