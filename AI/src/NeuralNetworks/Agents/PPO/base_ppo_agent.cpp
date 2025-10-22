#include "NeuralNetworks/Agents/PPO/base_ppo_agent.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/info_utils.hpp"

// Standard library includes
#include "Environment/quantum_circuit_environment.hpp"
#include "Utils/passes_utils.hpp"

#include <memory>
#include <utility>

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

BasePPOAgent::BasePPOAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {
  this->ppo_epsilon = GLOBAL_PARAMS["ppo_epsilon"].to_double();
}

void BasePPOAgent::set_rollout_data(const torch::Tensor &observations,
                                    const torch::Tensor &actions) {
  this->rollout_observations = observations.to(this->device);
  this->rollout_actions = actions.to(this->device);
}

std::pair<torch::Tensor, torch::Tensor> BasePPOAgent::get_losses(
    const torch::Tensor &rewards, const torch::Tensor &log_action_probs,
    const torch::Tensor &state_values, const torch::Tensor &entropy,
    double discount_factor, double gae_hyperparameter,
    double entropy_coefficient) {
  int T = rewards.size(0);
  torch::TensorOptions options = rewards.options();
  torch::Tensor advantages = torch::zeros({T}, options);
  torch::Tensor A_gae = torch::zeros({}, options);
  for (int t = T - 1; t >= 0; --t) {
    torch::Tensor delta_t =
        rewards[t] + discount_factor * state_values[t + 1] - state_values[t];
    A_gae = delta_t + discount_factor * gae_hyperparameter * A_gae;
    advantages[t] = A_gae;
  }
  torch::Tensor episode_obs = this->rollout_observations.slice(0, 0, T);
  torch::Tensor current_values = this->critic->forward(episode_obs);
  torch::Tensor targets = state_values.slice(0, 0, T) + advantages;
  torch::Tensor critic_loss = (current_values - targets.detach()).pow(2).mean();
  torch::Tensor current_action_probs = this->actor->forward(episode_obs);
  torch::Tensor current_log = current_action_probs.log();
  torch::Tensor current_log_action_probs =
      current_log.gather(-1, this->rollout_actions.unsqueeze(-1)).squeeze(-1);
  torch::Tensor ratio = torch::exp(current_log_action_probs - log_action_probs);
  torch::Tensor surr1 = ratio * advantages.detach();
  torch::Tensor surr2 =
      torch::clamp(ratio, 1.0f - this->ppo_epsilon, 1.0f + this->ppo_epsilon) *
      advantages.detach();
  torch::Tensor actor_surrogate = torch::minimum(surr1, surr2).mean();
  torch::Tensor current_entropy =
      -(current_action_probs * current_log).sum(-1).mean();
  torch::Tensor actor_loss =
      -actor_surrogate - entropy_coefficient * current_entropy;
  return {actor_loss, critic_loss};
}

void BasePPOAgent::save_model() const { BaseActorCritic::save_model(); }

std::vector<std::function<std::unique_ptr<Pass>()>>
BasePPOAgent::select_passes_for_circuit(const fs::path &circuit_path) {
  QuantumCircuitEnvironment environment(this->max_qubits);
  if (!environment.register_quantum_circuit(circuit_path)) {
    std::cerr << "Failed to register quantum circuit: " << circuit_path
              << std::endl;
    return {};
  }
  std::vector<std::function<std::unique_ptr<Pass>()>> selected_passes;
  bool keep_going = true;
  while (keep_going) {
    auto [action, _1, _2, _3] =
        this->select_action(environment.get_observation_as_torch_tensor());
    int action_index = action.item<int>();
    auto [_4, terminated, truncated] = environment.step(action_index);
    selected_passes.push_back(PASS_FUNCTIONS[action_index]);
    keep_going = !terminated && !truncated;
  }
  return selected_passes;
}

} // namespace ai_pass_selector