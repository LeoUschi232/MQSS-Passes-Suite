#ifndef AGENT_UTILS_HPP
#define AGENT_UTILS_HPP

#include "Cancellations.hpp"
#include "Decompositions.hpp"
#include "Transforms.hpp"

#include <memory>
#include <mlir/Pass/Pass.h>
#include <torch/torch.h>

using namespace mqss::opt;

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

/**
 *
 * @param index
 * @return
 */
std::pair<std::string, std::unique_ptr<mlir::Pass>>
getPassByIndex(unsigned int index);

/**
 *
 * @return
 */
unsigned int getNrOfPasses();

inline std::vector<std::function<std::unique_ptr<mlir::Pass>()> > passFunctions = {
    [] { return createZeroRxToIdPass(); },
    [] { return createZeroRyToIdPass(); },
    [] { return createZeroRzToIdPass(); },
    [] { return createCxCxToIdPass(); },
    [] { return createCyCyToIdPass(); },
    [] { return createCzCzToIdPass(); },
    [] { return createXXToIdPass(); },
    [] { return createYYToIdPass(); },
    [] { return createZZToIdPass(); },
    [] { return createSSdgToIdPass(); },
    [] { return createSdgSToIdPass(); },
    [] { return createTTdgToIdPass(); },
    [] { return createTdgTToIdPass(); },
    [] { return createHHToIdPass(); },
    [] { return createRxRxToRxPass(); },
    [] { return createRyRyToRyPass(); },
    [] { return createRzRzToRzPass(); },
    [] { return createHXHToZPass(); },
    [] { return createHZHToXPass(); },
    [] { return createXHZToHPass(); },
    [] { return createZHXToHPass(); },
    [] { return createHRxHToRzPass(); },
    [] { return createHRzHToRxPass(); },
    [] { return createHCxHToCzPass(); },
    [] { return createHCzHToCxPass(); },
    [] { return createHCrxHToCrzPass(); },
    [] { return createHCrzHToCrxPass(); },
    [] { return createXHToHZPass(); },
    [] { return createHXToZHPass(); },
    [] { return createYHToHYPass(); },
    [] { return createHYToYHPass(); },
    [] { return createZHToHXPass(); },
    [] { return createHZToXHPass(); },
    [] { return createSSSToSdgPass(); },
    [] { return createSdgSdgSdgToSPass(); },
    [] { return createSSToZPass(); },
    [] { return createSdgSdgToZPass(); },
    [] { return createSZToSdgPass(); },
    [] { return createZSToSdgPass(); },
    [] { return createSdgZToSPass(); },
    [] { return createZSdgToSPass(); },
    [] { return createTTToSPass(); },
    [] { return createCxCxCxToSwapPass(); },
    [] { return createCxZToZCxPass(); },
    [] { return createZCxToCxZPass(); },
    [] { return createCxXToXCxPass(); },
    [] { return createXCxToCxXPass(); },
    [] { return createCxRxToRxCxPass(); },
    [] { return createRxCxToCxRxPass(); },
    [] { return createReverseCxPass(); },
    [] { return createXToHZHPass(); },
    [] { return createZToHXHPass(); },
    [] { return createRxToHRzHPass(); },
    [] { return createRzToHRxHPass(); },
    [] { return createCxToUpperHCzHPass(); },
    [] { return createCxToLowerHCzHPass(); },
    [] { return createCzToUpperHCxHPass(); },
    [] { return createCzToLowerHCxHPass(); },
    [] { return createCrxToHCrzHPass(); },
    [] { return createCrzToHCrxHPass(); },
    [] { return createSdgToSSSPass(); },
    [] { return createSToSdgSdgSdgPass(); },
    [] { return createSToTTPass(); },
    [] { return createSwapToLowerCxCxCxPass(); },
    [] { return createSwapToUpperCxCxCxPass(); }
};


} // namespace ai_pass_selector

#endif // AGENT_UTILS_HPP