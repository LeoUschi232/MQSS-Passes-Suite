#include "NeuralNetworks/Agents/agent_utils.hpp"

// Mlir includes
#include "mlir_utils.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Torch includes
#include "torch/torch.h"

// Utils includes
#include "Utils/info_utils.hpp"

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
  AgentClass agent_class = AGENT_NAME_TO_CLASS.at(agent_attributes[0]);
  if (agent_attributes[1].rfind("mq", 0) != 0) {
    throw std::invalid_argument("Missing <mq> prefix");
  }
  unsigned int max_qubits =
      static_cast<unsigned int>(std::stoul(agent_attributes[1].substr(2)));
  return {agent_class, max_qubits, agent_attributes[2]};
}

std::unique_ptr<torch::optim::Optimizer>
makeOptimizer(OptimizerType optimizerType,
              const torch::nn::Sequential &agentModel, double learningRate) {
  switch (optimizerType) {
  case OptimizerType::Adagrad:
    return std::make_unique<torch::optim::Adagrad>(
        agentModel->parameters(), torch::optim::AdagradOptions(learningRate));
  case OptimizerType::Adam:
    return std::make_unique<torch::optim::Adam>(
        agentModel->parameters(), torch::optim::AdamOptions(learningRate));
  case OptimizerType::AdamW:
    return std::make_unique<torch::optim::AdamW>(
        agentModel->parameters(), torch::optim::AdamWOptions(learningRate));
  case OptimizerType::RMSProp:
    return std::make_unique<torch::optim::RMSprop>(
        agentModel->parameters(), torch::optim::RMSpropOptions(learningRate));
  case OptimizerType::SGD:
    return std::make_unique<torch::optim::SGD>(
        agentModel->parameters(), torch::optim::SGDOptions(learningRate));
  default:
    throw std::runtime_error("Unsupported optimizer type: " +
                             std::to_string(static_cast<int>(optimizerType)));
  }
}

unsigned int count_nr_trainable_parameters(const torch::nn::Module &network) {
  unsigned int total = 0;
  for (const torch::Tensor &param : network.parameters(true)) {
    if (param.requires_grad()) {
      total += static_cast<unsigned>(param.numel());
    }
  }
  return total;
}
} // namespace ai_pass_selector