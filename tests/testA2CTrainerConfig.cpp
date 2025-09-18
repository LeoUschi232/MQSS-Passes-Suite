#include <gtest/gtest.h>

#include "Torch/A2C/a2c_trainer.hpp"
#include "Torch/A2C/base_a2c_agent.hpp"
#include "Utils/circuit_utils.hpp"

#include <algorithm>
#include <unordered_map>

using namespace ai_pass_selector;

TEST(A2CTrainerConfigTest, AppliesOverrides) {
  std::unordered_map<std::string, std::string> params = {
      {"nr_parallel_environments", "7"},
      {"episodes", "3"},
      {"max_steps_per_episode", "11"},
      {"discount_factor", "0.5"},
      {"gae_hyperparameter", "0.77"},
      {"entropy_coefficient", "0.02"},
      {"device", "cpu"},
  };

  const auto config = build_a2c_trainer_config(params);

  EXPECT_EQ(config.nr_parallel_environments, 7u);
  EXPECT_EQ(config.episodes, 3u);
  EXPECT_EQ(config.max_steps_per_episode, 11u);
  EXPECT_DOUBLE_EQ(config.discount_factor, 0.5);
  EXPECT_DOUBLE_EQ(config.gae_hyperparameter, 0.77);
  EXPECT_DOUBLE_EQ(config.entropy_coefficient, 0.02);
  EXPECT_EQ(config.device.type(), torch::kCPU);

  for (const auto &expected_override : {
           std::string("nr_parallel_environments=7"),
           std::string("episodes=3"),
           std::string("max_steps_per_episode=11"),
           std::string("discount_factor=0.5"),
           std::string("gae_hyperparameter=0.77"),
           std::string("entropy_coefficient=0.02"),
           std::string("device=cpu"),
       }) {
    EXPECT_NE(
        std::find(
            config.applied_overrides.begin(),
            config.applied_overrides.end(),
            expected_override),
        config.applied_overrides.end())
        << "Missing override entry: " << expected_override;
  }
}

namespace {
class DummyAgent : public BaseA2CAgent {
public:
  DummyAgent() : BaseA2CAgent(TINY, {}) { nr_input_values = 1; }
  std::string agentName() const override { return "dummy"; }
};
} // namespace

TEST(A2CTrainerConfigTest, OverridesAffectTrainingFlow) {
  DummyAgent agent;
  std::unordered_map<std::string, std::string> params = {
      {"nr_parallel_environments", "0"},
  };

  const auto results = train_a2c(agent, "", params);
  EXPECT_TRUE(results.empty());
}
