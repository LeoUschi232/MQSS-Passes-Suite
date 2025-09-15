#ifndef AGENT_UTILS_HPP
#define AGENT_UTILS_HPP

#include <memory>
#include <torch/torch.h>

namespace ai_pass_selector {
constexpr int OPTIMIZER_ADAGRAD = 1;
constexpr int OPTIMIZER_ADAM = 2;
constexpr int OPTIMIZER_ADAMW = 3;
constexpr int OPTIMIZER_LBFGS = 4;
constexpr int OPTIMIZER_RMSPROP = 5;
constexpr int OPTIMIZER_SGD = 6;

/**
 *
 * @param optimizerType
 * @param agentModel
 * @param learningRate
 * @return
 */
std::unique_ptr<torch::optim::Optimizer> makeOptimizer(
    int optimizerType, const torch::nn::Sequential &agentModel,
    double learningRate);


} // namespace ai_pass_selector

#endif // AGENT_UTILS_HPP