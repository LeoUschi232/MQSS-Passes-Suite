#include "NeuralNetworks/Agents/abstract_agent.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/A2C/a2c_agents.hpp"
#include "NeuralNetworks/Agents/ACER/acer_agents.hpp"
#include "NeuralNetworks/Agents/PPO/ppo_agents.hpp"
#include "NeuralNetworks/Agents/SDSAC/sdsac_agents.hpp"
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/tensor_utils.hpp"

// Standard library includes
#include <memory>
#include <tuple>

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

AbstractAgent::AbstractAgent(unsigned int max_qubits)
    : max_qubits(std::max(max_qubits, GLOBAL_MIN_NR_QUBITS)) {
  this->nr_trainable_parmaeters = count_nr_trainable_parameters(*this);
}

unsigned int AbstractAgent::getMaxQubits() const { return this->max_qubits; }

unsigned int AbstractAgent::getNrTrainableParameters() const {
  return this->nr_trainable_parmaeters;
}

std::tuple<QuantumCircuit, int, int, std::vector<std::string>>
AbstractAgent::run_on_circuit(
    const fs::path &circuit_path,
    const std::vector<std::function<std::unique_ptr<Pass>()>> &pass_functions) {
  QuantumCircuit circuit(circuit_path);
  int initial_nr_gates = circuit.getNrGates();
  int initial_depth = circuit.getDepth();
  std::vector<std::string> pass_names;
  for (const std::function<std::unique_ptr<Pass>()> &pass_function :
       pass_functions) {
    auto pass_ptr = pass_function();
    auto name = std::string(pass_ptr.get()->getArgument());
    pass_names.push_back(name);
    auto [succeeded, wasApplied] = circuit.run_pass(pass_ptr);
    if (GLOBAL_PARAMS["print_diagnostics"].to_bool()) {
      if (!succeeded) {
        std::cerr << "Pass " << name << " failed on circuit " << circuit_path
                  << "." << std::endl;
        continue;
      }
      if (!wasApplied) {
        std::cout << "Pass " << name << " was not applied on circuit "
                  << circuit_path << "." << std::endl;
      }
    }
  }
  return {std::move(circuit),
          initial_nr_gates - static_cast<int>(circuit.getNrGates()),
          initial_depth - static_cast<int>(circuit.getDepth()), pass_names};
}

std::unique_ptr<AbstractAgent>
AbstractAgent::getAgent(const std::string &agent_name) {
  try {
    switch (AgentAttributes attributes = parseAgentName(agent_name);
            attributes.agent_class) {
    case AgentClass::A2C: {
      if (attributes.extras == "tcn") {
        return std::make_unique<A2C_TCN>(attributes.max_qubits);
      }
      if (attributes.extras == "lstm") {
        return std::make_unique<A2C_LSTM>(attributes.max_qubits);
      }
      if (attributes.extras == "hybrid") {
        return std::make_unique<A2C_HYBRID>(attributes.max_qubits);
      }
      std::cerr << "No such A2C agent: " << agent_name << std::endl;
      return {};
    }
    case AgentClass::PPO: {
      if (attributes.extras == "tcn") {
        return std::make_unique<PPO_TCN>(attributes.max_qubits);
      }
      if (attributes.extras == "lstm") {
        return std::make_unique<PPO_LSTM>(attributes.max_qubits);
      }
      if (attributes.extras == "hybrid") {
        return std::make_unique<PPO_HYBRID>(attributes.max_qubits);
      }
      std::cerr << "No such PPO agent: " << agent_name << std::endl;
      return {};
    }
    case AgentClass::SDSAC: {
      if (attributes.extras == "tcn") {
        return std::make_unique<SDSAC_TCN>(attributes.max_qubits);
      }
      if (attributes.extras == "lstm") {
        return std::make_unique<SDSAC_LSTM>(attributes.max_qubits);
      }
      if (attributes.extras == "hybrid") {
        return std::make_unique<SDSAC_HYBRID>(attributes.max_qubits);
      }
      std::cerr << "No such SDSAC agent: " << agent_name << std::endl;
      return {};
    }
    case AgentClass::ACER: {
      if (attributes.extras == "tcn") {
        return std::make_unique<ACER_TCN>(attributes.max_qubits);
      }
      if (attributes.extras == "lstm") {
        return std::make_unique<ACER_LSTM>(attributes.max_qubits);
      }
      if (attributes.extras == "hybrid") {
        return std::make_unique<ACER_HYBRID>(attributes.max_qubits);
      }
      std::cerr << "No such ACER agent: " << agent_name << std::endl;
      return {};
    }
    default:
      std::cerr << "No such agent yet: " << agent_name << std::endl;
      return {};
    }
  } catch (const std::runtime_error &error) {
    std::cerr << error.what() << std::endl;
  }
  return {};
}

} // namespace ai_pass_selector