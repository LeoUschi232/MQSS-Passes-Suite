#ifndef BASE_A3C_AGENT_HPP
#define BASE_A3C_AGENT_HPP

// Torch includes
#include "torch/torch.h"

// Standard library includes
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace fs = std::filesystem;

namespace ai_pass_selector {

class BaseA3CAgent : public torch::nn::Module {
protected:
  /// Attributes on configuration
  unsigned int max_qubits = 0u;
  int critic_optimizer_type = 0;
  int actor_optimizer_type = 0;
  double critic_learning_rate = 0.0;
  double actor_learning_rate = 0.0;
  torch::Device device = torch::kCPU;

  /// Global Attributes shared across all workers
  torch::nn::Sequential critic = nullptr;
  torch::nn::Sequential actor = nullptr;
  std::shared_ptr<torch::optim::Optimizer> actor_optimizer = nullptr;
  std::shared_ptr<torch::optim::Optimizer> critic_optimizer = nullptr;

  /// Mutex for thread safety
  std::unique_ptr<std::mutex> model_mutex = std::make_unique<std::mutex>();

  /// A3C specific attributes
  std::unordered_map<std::string, std::string> params_for_cloning = {};
  bool gradients_zero = true;
  bool is_boss = true;

public:
  /// Constructors
  BaseA3CAgent(unsigned int max_qubits,
               std::unordered_map<std::string, std::string> params,
               bool is_boss = true);

  /**
   *
   * @param params
   */
  void configure(std::unordered_map<std::string, std::string> params);

  /**
   *
   * @param actor
   * @param critic
   * @return
   */
  bool initialize(const torch::nn::Sequential &actor,
                  const torch::nn::Sequential &critic);

  /// Destructor
  ~BaseA3CAgent() override = default;

  /// Copy and move constructors and assignment operators
  BaseA3CAgent(const BaseA3CAgent &other) noexcept = delete;

  BaseA3CAgent(BaseA3CAgent &&other) noexcept = default;

  BaseA3CAgent &operator=(const BaseA3CAgent &other) noexcept = delete;

  BaseA3CAgent &operator=(BaseA3CAgent &&other) noexcept = delete;

  /// Getters
  unsigned int getMaxQubits() const;

  //////////////////////////////////////////////////////////////////////////////
  /// A2C standard methods
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
   * @param observation
   * @return
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  select_action(const torch::Tensor &observation);

  /**
   * No termination masks because A3C uses asynchronous worker agents, each of
   * which has just 1 environment instance instead of a synchronous agent with a
   * batch of environments.
   * @param rewards
   * @param log_action_probs
   * @param state_values
   * @param entropy
   * @param discount_factor
   * @param gae_hyperparameter
   * @param entropy_coefficient
   * @return
   */
  static std::pair<torch::Tensor, torch::Tensor>
  get_losses(const torch::Tensor &rewards,
             const torch::Tensor &log_action_probs,
             const torch::Tensor &state_values, const torch::Tensor &entropy,
             double discount_factor, double gae_hyperparameter,
             double entropy_coefficient);

  /**
   *
   * @param actor_loss
   * @param critic_loss
   */
  void update_parameters(const torch::Tensor &actor_loss,
                         const torch::Tensor &critic_loss) const;
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  /// A3C specific methods for worker concurrency
  void zero_grad();

  /**
   * Necessary to create worker agents for asynchronous training.
   * @return
   */
  virtual std::unique_ptr<BaseA3CAgent> clone() const = 0;

  void load_params(BaseA3CAgent &other);
  void load_gradients(BaseA3CAgent &other);
  void update_parameters_assuming_gradients_are_loaded();
  //////////////////////////////////////////////////////////////////////////////

  /// Saving and Loading
  void save_model() const;
  void load_model();
  virtual std::string agentName() const = 0;
};
} // namespace ai_pass_selector

#endif // BASE_A3C_AGENT_HPP
