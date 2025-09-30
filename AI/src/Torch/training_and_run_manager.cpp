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
      if (attributes.extras == "conv2npi") {
        agent = std::make_unique<A2C_CONV2NPI>(attributes.max_qubits, params);
      } else if (attributes.extras == "conv3npi") {
        agent = std::make_unique<A2C_CONV3NPI>(attributes.max_qubits, params);
      } else if (attributes.extras == "conv4npi") {
        agent = std::make_unique<A2C_CONV4NPI>(attributes.max_qubits, params);
      } else if (attributes.extras == "conv2nfull") {
        agent = std::make_unique<A2C_CONV2NFULL>(attributes.max_qubits, params);
      } else if (attributes.extras == "conv3nfull") {
        agent = std::make_unique<A2C_CONV3NFULL>(attributes.max_qubits, params);
      } else if (attributes.extras == "conv4nfull") {
        agent = std::make_unique<A2C_CONV4NFULL>(attributes.max_qubits, params);
      } else {
        std::cerr << "No such A2C agent yet: " << agent_name << std::endl;
        return {};
      }
      agent->load_model();
      training_results = train_a2c(*agent, dataset, params);
      break;
    }
    case A3C:
    case PPO:
    default:
      std::cerr << "No such A2C agent yet: " << agent_name << std::endl;
      return {};
    }
  } catch (const std::runtime_error &e) {
    std::cerr << "\n" << e.what() << std::endl;
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