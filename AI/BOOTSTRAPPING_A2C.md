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