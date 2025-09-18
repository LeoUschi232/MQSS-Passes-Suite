#include "Torch/training_and_run_manager.hpp"

// Environment includes
#include <Environment/environment.hpp>

// Torch includes
#include <Torch/A2C/base_a2c_agent.hpp>

// Utils includes
#include <Utils/passes_utils.hpp>

// Stdandard library includes
#include <unordered_map>
#include <string>
#include <Torch/A2C/a2c_ib_fc_lsd.hpp>
#include <Torch/A2C/a2c_ib_fc_lsm.hpp>
#include <Utils/circuit_utils.hpp>
#include <Utils/info_utils.hpp>

namespace fs = std::filesystem;

namespace ai_pass_selector {
std::unordered_map<std::string, std::string> train_agent(
    const std::string &agent_name,
    const std::string &dataset,
    std::unordered_map<std::string, std::string> params) {
  std::unordered_map<std::string, std::string> training_results;

  switch (AgentAttributes attributes = parseAgentName(agent_name);
    attributes.agent_class) {
  case A2C:
    break;
  case A3C:
    break;
  case PPO:
    break;
  case RNN:
    break;
  }
  return training_results;
}


std::unordered_map<std::string, std::string> run(
    const std::string &agent_name,
    const std::string &circuit,
    const std::string &output,
    std::unordered_map<std::string, std::string> params) {
  throw std::runtime_error("Not implemented yet");
}

} // namespace ai_pass_selector