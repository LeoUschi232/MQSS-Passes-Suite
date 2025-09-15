#include "Environment/quantum_circuit_tensor.hpp"
#include "Torch/agent_utils..hpp"

#include <torch/torch.h>

namespace ai_pass_selector {

std::unique_ptr<torch::optim::Optimizer> makeOptimizer(
    int optimizerType, const torch::nn::Sequential &agentModel,
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
    throw std::runtime_error(
        "Unsupported optimizer type: " + std::to_string(optimizerType));
  }
}


std::pair<std::string, std::unique_ptr<mlir::Pass> >
getPassByIndex(unsigned int index) {
  if (index >= getNrOfPasses()) {
    std::cerr << "In getPassByIndex: " << index << std::endl;
    return {"", nullptr};
  }
  std::unique_ptr<mlir::Pass> pass = passFunctions[index]();
  return {std::string(pass.get()->getArgument()), std::move(pass)};
}

unsigned int getNrOfPasses() {
  return passFunctions.size();
}

unsigned int getNrOfInputValuesForInstructionBased(
    unsigned int max_qubits, unsigned int max_instructions) {
  return max_instructions * (max_qubits + NR_GATES + MAX_GATE_PARAMS);
}

unsigned int getNrOfInputValuesForDepthBased(
    unsigned int max_qubits, unsigned int max_depth) {
  return max_depth * max_qubits
         * (NR_GATES + MAX_GATE_PARAMS + QUBIT_ROLE + max_qubits);
}
} // namespace ai_pass_selector