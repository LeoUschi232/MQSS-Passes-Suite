#include "Torch/agent_utils.hpp"

#include <mlir_utils.hpp>
#include <Utils/circuit_utils.hpp>
#include <torch/torch.h>

using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {

int mapToOptimizerType(std::string &optimizer_name) {
  std::transform(
      optimizer_name.begin(), optimizer_name.end(), optimizer_name.begin(),
      [](unsigned char c) { return std::tolower(c); });
  if (optimizer_name == "adagrad") {
    return OPTIMIZER_ADAGRAD;
  }
  if (optimizer_name == "adam") {
    return OPTIMIZER_ADAM;
  }
  if (optimizer_name == "adamw") {
    return OPTIMIZER_ADAMW;
  }
  if (optimizer_name == "lbfgs") {
    return OPTIMIZER_LBFGS;
  }
  if (optimizer_name == "rmsprop") {
    return OPTIMIZER_RMSPROP;
  }
  if (optimizer_name == "sgd") {
    return OPTIMIZER_SGD;
  }
  // Default to Adam if unknown
  return OPTIMIZER_ADAM;
}

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

std::string select_best_agent(const std::string &circuit) {
  auto [module, contex_ptr] = extractMLIRContext(circuit);
  switch (auto [nrQubits, nrGates, depth]
        = getQubitsInstructionsDepth(FuncOp(module));
    classify_circuit(nrQubits, nrGates, depth)) {
  case TINY:
    return "a2c-ib-fc-lsd-5x25x10";
  case SMALL:
    return "a2c-ib-fc-lsd-20x500x50";
  case MODERATE:
    return "a2c-ib-fc-lsd-100x1000x200";
  case BIG:
  case HUGE:
  default:
    break;
  }
  throw std::runtime_error("No suitable agent found for circuit.");
}


std::tuple<std::vector<std::string>, std::vector<std::unique_ptr<mlir::Pass> > >
getRecommendedPasses(const std::string &agent_name,
                     const std::string &circuit) {

}
} // namespace ai_pass_selector