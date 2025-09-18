#include "Torch/A2C/base_a2c_agent.hpp"

// Torch includes
#include <torch/torch.h>

// Utils includes
#include "Torch/agent_utils.hpp"

// Standard library includes
#include <utility>
#include <tuple>
#include <cmath>
#include <memory>
#include <Utils/circuit_utils.hpp>

namespace ai_pass_selector {
BaseA2CAgent::BaseA2CAgent(
    int circuit_size_class,
    std::unordered_map<std::string, std::string> params) {
  this->configure(circuit_size_class, std::move(params));
}

BaseA2CAgent::BaseA2CAgent(
    const std::string &circuit_size,
    std::unordered_map<std::string, std::string> params) {
  if (CIRCUIT_SIZE_TO_CLASS.find(circuit_size)
      == CIRCUIT_SIZE_TO_CLASS.end()) {
    throw std::runtime_error("Unsupported size: " + circuit_size);
  }
  int circuit_size_class = CIRCUIT_SIZE_TO_CLASS.at(circuit_size);
  this->configure(circuit_size_class, std::move(params));
}

void BaseA2CAgent::configure(
    int circuit_size_class,
    std::unordered_map<std::string, std::string> params) {
  if (CIRCUIT_CLASS_TO_SPECS.find(circuit_size_class)
      == CIRCUIT_CLASS_TO_SPECS.end()) {
    throw std::runtime_error("Unsupported size class: " + circuit_size_class);
  }
  this->size_class = circuit_size_class;
  std::tie(this->max_qubits, this->max_instructions, this->max_depth)
      = CIRCUIT_CLASS_TO_SPECS.at(circuit_size_class);
  for (auto [key, value] : params) {
    if (key == "critic_optimizer") {
      if (OPTIMIZER_NAME_TO_TYPE.find(value) == OPTIMIZER_NAME_TO_TYPE.end()) {
        throw std::runtime_error("Unsupported optimizer: " + value);
      }
      this->critic_optimizer_type = OPTIMIZER_NAME_TO_TYPE.at(value);
    } else if (key == "actor_optimizer") {
      if (OPTIMIZER_NAME_TO_TYPE.find(value) == OPTIMIZER_NAME_TO_TYPE.end()) {
        throw std::runtime_error("Unsupported optimizer: " + value);
      }
      this->actor_optimizer_type = OPTIMIZER_NAME_TO_TYPE.at(value);
    } else if (key == "critic_learning_rate") {
      this->critic_learning_rate = std::stod(value);
    } else if (key == "actor_learning_rate") {
      this->actor_learning_rate = std::stod(value);
    } else if (key == "nr_parallel_environments") {
      this->nr_parallel_environments = std::stoul(value);
    } else if (key == "device"
               && (value == "cuda" || value == "gpu")
               && torch::cuda::is_available()) {
      this->device = torch::kCUDA;
    }
  }
}


bool BaseA2CAgent::initialize(
    int nr_input_values,
    const torch::nn::Sequential &critic,
    const torch::nn::Sequential &actor) {
  try {
    this->nr_input_values = nr_input_values;
    this->critic = critic;
    this->actor = actor;
    register_module("critic", this->critic);
    register_module("actor", this->actor);
    this->critic->to(this->device);
    this->actor->to(this->device);
    this->critic_optimizer = makeOptimizer(
        critic_optimizer_type, this->critic, this->critic_learning_rate);
    this->actor_optimizer = makeOptimizer(
        actor_optimizer_type, this->actor, this->actor_learning_rate);
  } catch (const std::runtime_error &e) {
    this->nr_input_values = 0;
    std::cerr << e.what() << std::endl;
    return false;
  }
  return true;
}

unsigned int BaseA2CAgent::getMaxQubits() const {
  return this->max_qubits;
}

unsigned int BaseA2CAgent::getMaxInstructions() const {
  return this->max_instructions;
}

unsigned int BaseA2CAgent::getMaxDepth() const {
  return this->max_depth;
}

unsigned int BaseA2CAgent::getNrInputValues() const {
  return this->nr_input_values;
}

std::pair<torch::Tensor, torch::Tensor> BaseA2CAgent::forward(
    torch::Tensor batched_observations) {
  batched_observations = batched_observations.to(this->device);
  return {this->critic->forward(batched_observations),
          this->actor->forward(batched_observations)};
}

std::tuple<std::vector<unsigned int>,
           torch::Tensor, torch::Tensor, torch::Tensor>
BaseA2CAgent::select_action(const torch::Tensor &batched_observations) {
  auto [state_values, action_probs] = this->forward(batched_observations);
  // sample one action per row; result is [B,1] -> squeeze to [B]
  torch::Tensor actions_tensor = action_probs.multinomial(1).squeeze(-1);
  // CUDA tensors can’t be read directly)
  torch::Tensor actions_cpu = actions_tensor.to(torch::kCPU);
  std::vector<unsigned int> actions;
  actions.reserve(actions_cpu.size(0));
  for (int64_t i = 0; i < actions_cpu.size(0); ++i) {
    actions.push_back(static_cast<unsigned int>(
      actions_cpu[i].item<int64_t>()));
  }

  // log π(a|s) for the sampled actions: gather along the action dim
  const torch::Tensor log_action_probs = action_probs.log();
  // a_t, log π(a_t|s_t), V(s_t), entropy of π(a_t|s_t)
  return {
      actions,
      log_action_probs.gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1),
      state_values.squeeze(-1),
      -(action_probs * log_action_probs).sum(-1)
  };
}

std::pair<torch::Tensor, torch::Tensor> BaseA2CAgent::get_losses(
    const torch::Tensor &rewards,
    const torch::Tensor &log_action_probs,
    const torch::Tensor &state_values,
    const torch::Tensor &entropy,
    const torch::Tensor &termination_masks,
    const double discount_factor,
    const double gae_hyperparameter,
    const double entropy_coefficient) {

  // Let T = final timestep of an episode.
  // An episode generates T+1 states from S_0 to S_T.
  // An episode generates T actions from A_0 to A_{T-1}.
  // An episode generates T rewards from R_1 to R_T.
  int T = rewards.size(0);
  int B = rewards.size(1);
  const torch::TensorOptions options = rewards.options();
  torch::Tensor advantages = torch::zeros({T, B}, options);

  // Compute the advantages using Generalized Advantage Estimation.
  // Temporal Difference is a method used in Reinforcement Learning to estimate the value function of a state
  // based on the difference between the immediate reward obtained from a current state
  // and the estimated value of the next state.
  torch::Tensor A_gae = torch::zeros({B}, options);
  const torch::Tensor zero_state_value = torch::zeros({B}, state_values.options());
  for (int t = T - 1; t >= 0; --t) {
    const torch::Tensor next_state_value
        = (t + 1 < state_values.size(0)) ? state_values[t + 1] : zero_state_value;

    // In Barto & Sutton the temporal difference residual of V with discount gamma is:
    // delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    torch::Tensor delta_t
        = rewards[t] - state_values[t]
          + discount_factor * next_state_value * termination_masks[t];

    // The generalized advantage estimation defined by Schulman et al is:
    // A_gae = sum_{l=0}^{\infty} (gamma * lamda)^l * delta_{t+l}
    A_gae = discount_factor * gae_hyperparameter * A_gae * termination_masks[t]
            + delta_t;
    advantages[t] = A_gae;
  }

  // The equation for the Value function performance measure is:
  // J(w) = (1/N) * sum_{t=0}^{N-1} (A(s_t, a_t)^2)
  auto critic_loss = advantages.pow(2).mean();

  // Give a bonus for higher entropy to encourage exploration.
  // The equation for the policy performance measure is:
  // J(θ) = (1/N) * sum_{t=0}^{N-1} (ln π_θ(a_t|s_t) * A(s_t, a_t))
  auto actor_loss = -(log_action_probs * advantages.detach()).mean()
                    - entropy_coefficient * entropy.mean();
  return {critic_loss, actor_loss};
}

void BaseA2CAgent::update_parameters(
    const torch::Tensor &critic_loss,
    const torch::Tensor &actor_loss) const {
  std::lock_guard lock(*model_mutex);
  this->critic_optimizer->zero_grad();
  critic_loss.backward();
  this->critic_optimizer->step();
  this->actor_optimizer->zero_grad();
  actor_loss.backward();
  this->actor_optimizer->step();
}

void BaseA2CAgent::save_model() const {
  std::lock_guard lock(*model_mutex);
  if (this->nr_input_values <= 0) {
    std::cerr << "No agent to save." << std::endl;
    return;
  }
  std::string name = this->agentName();
  if (name.empty()) {
    std::cerr << "No agent to save." << std::endl;
    return;
  }
  std::string critic_path = std::string(AI_AGENTS_DIR) + name + "-critic.pt";
  std::string actor_path = std::string(AI_AGENTS_DIR) + name + "-actor.pt";
  torch::save(this->critic, critic_path);
  torch::save(this->actor, actor_path);
}

void BaseA2CAgent::load_model() {
  std::lock_guard lock(*model_mutex);
  if (this->nr_input_values <= 0) {
    std::cerr << "No agent to load." << std::endl;
    return;
  }
  std::string name = this->agentName();
  if (name.empty()) {
    std::cerr << "No agent to save." << std::endl;
    return;
  }
  std::string critic_path = std::string(AI_AGENTS_DIR) + name + "-critic.pt";
  std::string actor_path = std::string(AI_AGENTS_DIR) + name + "-actor.pt";
  torch::load(this->critic, critic_path);
  torch::load(this->actor, actor_path);
}

} // ai_pass_selector