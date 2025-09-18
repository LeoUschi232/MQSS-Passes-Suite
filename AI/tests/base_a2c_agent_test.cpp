#include <Torch/A2C/base_a2c_agent.hpp>
#include <Torch/agent_utils.hpp>
#include <Utils/circuit_utils.hpp>

#include <torch/torch.h>

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>

namespace ai_pass_selector {
namespace {

class DummyA2CAgent : public BaseA2CAgent {
  using BaseA2CAgent::BaseA2CAgent;

public:
  std::string agentName() const override { return "dummy"; }

  int getCriticOptimizerType() const { return this->critic_optimizer_type; }
  int getActorOptimizerType() const { return this->actor_optimizer_type; }
  double getCriticLearningRate() const { return this->critic_learning_rate; }
  double getActorLearningRate() const { return this->actor_learning_rate; }
  torch::optim::Optimizer *getCriticOptimizer() const {
    return this->critic_optimizer.get();
  }
  torch::optim::Optimizer *getActorOptimizer() const {
    return this->actor_optimizer.get();
  }
};

bool doublesEqual(double lhs, double rhs, double eps = 1e-12) {
  return std::fabs(lhs - rhs) <= eps;
}

} // namespace
} // namespace ai_pass_selector

int main() {
  using namespace ai_pass_selector;

  DummyA2CAgent default_agent(TINY, {});
  if (default_agent.getCriticOptimizerType() != OPTIMIZER_ADAM) {
    return 1;
  }
  if (default_agent.getActorOptimizerType() != OPTIMIZER_ADAM) {
    return 2;
  }
  if (!doublesEqual(default_agent.getCriticLearningRate(), 1e-3)) {
    return 3;
  }
  if (!doublesEqual(default_agent.getActorLearningRate(), 1e-3)) {
    return 4;
  }

  constexpr int nr_inputs = 4;
  auto critic = torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(nr_inputs, 1)));
  auto actor = torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(nr_inputs, 2)));
  if (!default_agent.initialize(nr_inputs, critic, actor)) {
    return 5;
  }
  if (default_agent.getCriticOptimizer() == nullptr) {
    return 6;
  }
  if (default_agent.getActorOptimizer() == nullptr) {
    return 7;
  }

  std::unordered_map<std::string, std::string> overrides = {
      {"critic_optimizer", "sgd"},
      {"actor_optimizer", "rmsprop"},
      {"critic_learning_rate", "0.005"},
      {"actor_learning_rate", "0.007"}};
  DummyA2CAgent overridden_agent(TINY, overrides);
  if (overridden_agent.getCriticOptimizerType() != OPTIMIZER_SGD) {
    return 8;
  }
  if (overridden_agent.getActorOptimizerType() != OPTIMIZER_RMSPROP) {
    return 9;
  }
  if (!doublesEqual(overridden_agent.getCriticLearningRate(), 0.005)) {
    return 10;
  }
  if (!doublesEqual(overridden_agent.getActorLearningRate(), 0.007)) {
    return 11;
  }

  auto critic_overridden = torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(nr_inputs, 1)));
  auto actor_overridden = torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(nr_inputs, 2)));
  if (!overridden_agent.initialize(nr_inputs, critic_overridden,
                                   actor_overridden)) {
    return 12;
  }

  if (dynamic_cast<torch::optim::SGD *>(
          overridden_agent.getCriticOptimizer()) == nullptr) {
    return 13;
  }
  if (dynamic_cast<torch::optim::RMSprop *>(
          overridden_agent.getActorOptimizer()) == nullptr) {
    return 14;
  }

  return 0;
}
