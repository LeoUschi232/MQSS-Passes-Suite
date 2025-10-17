#include "NeuralNetworks/Agents/training_and_run_manager.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Torch includes
#include "NeuralNetworks/Agents/A3C/a3c_agents.hpp"
#include "NeuralNetworks/Agents/A3C/a3c_trainer.hpp"
#include "NeuralNetworks/Agents/A3C/base_a3c_agent.hpp"
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"

// Stdandard library includes
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

std::unordered_map<std::string, std::string>
train(const std::string &agent_name, const std::string &dataset) {
  std::unordered_map<std::string, std::string> training_results;
  try {
    switch (AgentAttributes attributes = parseAgentName(agent_name);
            attributes.agent_class) {
    case AgentClass::A3C: {
      std::unique_ptr<BaseA3CAgent> agent;
      if (attributes.extras == "tcnrelu") {
        agent = std::make_unique<A3C_TCN_RELU>(attributes.max_qubits);
      } else if (attributes.extras == "tcnprelu") {
        agent = std::make_unique<A3C_TCN_PRELU>(attributes.max_qubits);
      } else if (attributes.extras == "lstmhmpp") {
        agent = std::make_unique<A3C_LSTM_HMPP>(attributes.max_qubits);
      } else if (attributes.extras == "lstmbmnp") {
        agent = std::make_unique<A3C_LSTM_BMNP>(attributes.max_qubits);
      } else if (attributes.extras == "hybrid") {
        agent = std::make_unique<A3C_HYBRID>(attributes.max_qubits);
      } else {
        std::cerr << "No such A3C agent: " << agent_name << std::endl;
        return {};
      }
      agent->load_model();
      unsigned int nr_asynchronous_agents =
          GLOBAL_PARAMS["nr_asynchronous_agents"].to_int();
      if (nr_asynchronous_agents <= 1u) {
        std::cout << "Only 1 asnc A3C agent => Defaulting to A2C training."
                  << std::endl;
        training_results = train_a2c(agent, dataset);
      } else {
        training_results = train_a3c(agent, dataset);
      }
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

std::unordered_map<std::string, std::string> run(const std::string &agent_name,
                                                 const std::string &circuit,
                                                 const std::string &output) {
  throw std::runtime_error("Not implemented yet");
}

} // namespace ai_pass_selector