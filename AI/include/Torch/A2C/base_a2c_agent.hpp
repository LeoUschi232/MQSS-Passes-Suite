#ifndef BASE_A2C_AGENT_HPP
#define BASE_A2C_AGENT_HPP

// Torch includes
#include <torch/torch.h>

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
  int size_class = 0;
  unsigned int max_qubits = 0;
  int critic_optimizer_type = 0;
  int actor_optimizer_type = 0;
  double critic_learning_rate = 0;
  double actor_learning_rate = 0;
  unsigned int nr_parallel_environments = 0;
  torch::Device device = torch::kCPU;

  /// Attributes on initialization
  unsigned int nr_input_values = 0;
  torch::nn::Sequential critic = nullptr;
  torch::nn::Sequential actor = nullptr;
  std::unique_ptr<torch::optim::Optimizer> actor_optimizer = nullptr;
  std::unique_ptr<torch::optim::Optimizer> critic_optimizer = nullptr;

  /// Mutex for thread safety
  std::unique_ptr<std::mutex> model_mutex = std::make_unique<std::mutex>();

public:
  /// Constructors
  BaseA2CAgent(int circuit_size_class,
               std::unordered_map<std::string, std::string> params);

  BaseA2CAgent(const std::string &circuit_size,
               std::unordered_map<std::string, std::string> params);

  /**
   *
   * @param circuit_size_class
   * @param params
   */
  void configure(int circuit_size_class,
                 std::unordered_map<std::string, std::string> params);

  /**
   *
   * @param nr_input_values
   * @param critic
   * @param actor
   * @return
   */
  bool initialize(int nr_input_values, const torch::nn::Sequential &critic,
                  const torch::nn::Sequential &actor);

  /// Destructor
  ~BaseA2CAgent() override = default;

  /// Copy and move constructors and assignment operators
  BaseA2CAgent(const BaseA2CAgent &other) = delete;

  BaseA2CAgent(BaseA2CAgent &&other) noexcept = default;

  BaseA2CAgent &operator=(const BaseA2CAgent &other) = delete;

  BaseA2CAgent &operator=(BaseA2CAgent &&other) noexcept = delete;

  /// Getters
  unsigned int getMaxQubits() const;

  unsigned int getMaxInstructions() const;

  unsigned int getMaxDepth() const;

  unsigned int getNrInputValues() const;

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
   * @return
   */
  static std::pair<torch::Tensor, torch::Tensor>
  get_losses(const torch::Tensor &rewards,
             const torch::Tensor &log_action_probs,
             const torch::Tensor &state_values, const torch::Tensor &entropy,
             const torch::Tensor &termination_masks, double discount_factor,
             double gae_hyperparameter, double entropy_coefficient);

  /**
   *
   * @param critic_loss
   * @param actor_loss
   */
  void update_parameters(const torch::Tensor &critic_loss,
                         const torch::Tensor &actor_loss) const;

  /**
   *
   */
  void save_model() const;

  /**
   *
   */
  void load_model();

  virtual std::string agentName() const = 0;
};
} // namespace ai_pass_selector

#endif // BASE_A2C_AGENT_HPP
