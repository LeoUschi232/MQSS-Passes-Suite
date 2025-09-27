#include "Torch/training_and_run_manager.hpp"

// Environment includes
#include "Environment/environment.hpp"

// Torch includes
#include "Torch/A2C/a2c_agents.hpp"
#include "Torch/A2C/a2c_trainer.hpp"
#include "Torch/A2C/base_a2c_agent.hpp"
#include "Torch/agent_utils.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"

// Stdandard library includes
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

namespace ai_pass_selector {
std::unordered_map<std::string, std::string>
train(const std::string &agent_name, const std::string &dataset,
      std::unordered_map<std::string, std::string> params) {
  std::unordered_map<std::string, std::string> training_results;
  try {
    switch (AgentAttributes attributes = parseAgentName(agent_name);
            attributes.agent_class) {
    case A2C: {
      std::unique_ptr<BaseA2CAgent> agent;
      if (attributes.specifier == "ibfclsd") {
        agent = std::make_unique<A2C_IBCONV2>(attributes.size_class, params);
      } else if (attributes.specifier == "ibfclsm") {
        agent = std::make_unique<A2C_IBCONV4>(attributes.size_class, params);
      } else {
        throw std::runtime_error("Unknown A2C specifier: " +
                                 attributes.specifier);
      }
      agent->load_model();
      training_results = train_a2c(*agent, dataset, params);
      break;
    }
    case A3C:
    case PPO:
    case RNN:
    default:
      throw std::runtime_error("Unsupported agent_class in train()");
    }
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return {};
  }
  return training_results;
}

std::unordered_map<std::string, std::string>
run(const std::string &agent_name, const std::string &circuit,
    const std::string &output,
    std::unordered_map<std::string, std::string> params) {
  throw std::runtime_error("Not implemented yet");
}

} // namespace ai_pass_selector