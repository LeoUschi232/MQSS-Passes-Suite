#include "Agents/training_and_run_manager.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Torch includes
#include "Agents/A3C/a3c_agents.hpp"
#include "Agents/A3C/a3c_trainer.hpp"
#include "Agents/A3C/base_a3c_agent.hpp"
#include "Agents/agent_utils.hpp"

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
    case A3C: {
      std::unique_ptr<BaseA3CAgent> agent;
      if (attributes.extras == "conv2") {
        agent = std::make_unique<A3C_CONV2>(attributes.max_qubits, params);
      } else {
        std::cerr << "No such A3C agent yet: " << agent_name << std::endl;
        return {};
      }
      agent->load_model();
      training_results = train_a3c(std::move(agent), dataset, params);
      break;
    }
    default:
      std::cerr << "No such agent yet: " << agent_name << std::endl;
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