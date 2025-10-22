#include "NeuralNetworks/Agents/abstract_agent.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Neural-Networks includes
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

unsigned int AbstractAgent::getMaxQubits() const { return this->max_qubits; }

std::tuple<QuantumCircuit, unsigned int, unsigned int>
AbstractAgent::run_on_circuit(
    const fs::path &circuit_path,
    const std::vector<std::function<std::unique_ptr<Pass>()>> &pass_functions) {
  QuantumCircuit circuit(circuit_path);
  unsigned int initial_nr_gates = circuit.getNrGates();
  unsigned int initial_depth = circuit.getDepth();
  for (const std::function<std::unique_ptr<Pass>()> &pass_function :
       pass_functions) {
    auto pass_ptr = pass_function();
    auto name = std::string(pass_ptr.get()->getArgument());
    auto [succeeded, wasApplied] = circuit.run_pass(pass_ptr);
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
  return {std::move(circuit), initial_nr_gates - circuit.getNrGates(),
          initial_depth - circuit.getDepth()};
}
} // namespace ai_pass_selector