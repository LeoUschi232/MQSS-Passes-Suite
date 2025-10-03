#include "Agents/agent_utils.hpp"

// Mlir includes
#include "mlir_utils.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/info_utils.hpp"
#include "Utils/passes_utils.hpp"

using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {

AgentAttributes parseAgentName(const std::string &agent_name) {
  std::vector<std::string> agent_attributes = split_string(agent_name, '-');
  if (agent_attributes.size() != 3) {
    throw std::runtime_error("Invalid agent name: " + agent_name);
  }
  // The first agent attribute is the agent classe.
  // The second agent attribute is the size class.
  // The rest are agent-specific attributes.
  if (AGENT_NAME_TO_CLASS.find(agent_attributes[0]) ==
      AGENT_NAME_TO_CLASS.end()) {
    throw std::runtime_error("Unsupported agent: " + agent_attributes[0]);
  }
  int agent_class = AGENT_NAME_TO_CLASS.at(agent_attributes[0]);
  if (agent_attributes[1].rfind("mq", 0) != 0) {
    throw std::invalid_argument("Missing <mq> prefix");
  }
  unsigned int max_qubits =
      static_cast<unsigned int>(std::stoul(agent_attributes[1].substr(2)));
  return {agent_class, max_qubits, agent_attributes[2]};
}

std::unique_ptr<torch::optim::Optimizer>
makeOptimizer(int optimizerType, const torch::nn::Sequential &agentModel,
              double learningRate) {
  switch (optimizerType) {
  case OPTIMIZER_ADAGRAD:
    return std::make_unique<torch::optim::Adagrad>(
        agentModel->parameters(), torch::optim::AdagradOptions(learningRate));
  case OPTIMIZER_ADAM:
    return std::make_unique<torch::optim::Adam>(
        agentModel->parameters(), torch::optim::AdamOptions(learningRate));
  case OPTIMIZER_ADAMW:
    return std::make_unique<torch::optim::AdamW>(
        agentModel->parameters(), torch::optim::AdamWOptions(learningRate));
  case OPTIMIZER_LBFGS:
    return std::make_unique<torch::optim::LBFGS>(
        agentModel->parameters(), torch::optim::LBFGSOptions(learningRate));
  case OPTIMIZER_RMSPROP:
    return std::make_unique<torch::optim::RMSprop>(
        agentModel->parameters(), torch::optim::RMSpropOptions(learningRate));
  case OPTIMIZER_SGD:
    return std::make_unique<torch::optim::SGD>(
        agentModel->parameters(), torch::optim::SGDOptions(learningRate));
  default:
    throw std::runtime_error("Unsupported optimizer type: " +
                             std::to_string(optimizerType));
  }
}

std::string select_best_agent(const std::string &circuit) {
  std::cout << "Selecting best agent for circuit: " << circuit << std::endl;
  auto found_circuit = search_circuit(circuit);
  if (!found_circuit.has_value()) {
    throw std::runtime_error("Circuit '" + circuit + "' could not be found.");
  }

  auto circuit_path = found_circuit.value();
  if (circuit_path.extension().string() != ".qke") {
    throw std::runtime_error("Fail 1: " + circuit_path.stem().string());
  }

  std::string quake_module_text = readFileToString(circuit_path.string());
  if (quake_module_text.empty()) {
    throw std::runtime_error("Fail 2: " + circuit_path.stem().string());
  }

  auto [module, context_ptr] = extractMLIRContext(quake_module_text);
  // TODO: Do like check for agents
  throw std::runtime_error("No suitable agent found for circuit.");
}

std::tuple<std::vector<std::string>, std::vector<unsigned int>>
getRecommendedPasses(const std::string &agent_name, const std::string &circuit,
                     unsigned int nr_passes, fs::path output_path) {
  if (agent_name.empty()) {
    std::cerr << "No agent specified for recommending passes." << std::endl;
    return {};
  }
  auto found_circuit = search_circuit(circuit);
  if (!found_circuit.has_value()) {
    std::cerr << "Failed to get circuit: " << circuit << std::endl;
    return {};
  }
  if (auto circuit_path = found_circuit.value(); circuit_path.empty()) {
    return {};
  }
  std::vector<std::unique_ptr<mlir::Pass>> passes;
  std::vector<std::string> pass_names;
  std::vector<unsigned int> pass_indexes;

  std::vector<std::string> agent_attributes = split_string(agent_name, '-');
  if (AGENT_NAME_TO_CLASS.find(agent_attributes[0]) ==
      AGENT_NAME_TO_CLASS.end()) {
    std::cerr << "Unsupported agent: " << agent_attributes[0] << std::endl;
    return {};
  }
  int agent_class = AGENT_NAME_TO_CLASS.at(agent_attributes[0]);
  if (agent_attributes[1].rfind("mq", 0) != 0) {
    throw std::invalid_argument("Missing 'mq' prefix");
  }
  unsigned int max_qubits =
      static_cast<unsigned int>(std::stoul(agent_attributes[1].substr(2)));
  return {};
}
} // namespace ai_pass_selector