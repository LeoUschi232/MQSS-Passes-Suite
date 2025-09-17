#ifndef AGENT_UTILS_HPP
#define AGENT_UTILS_HPP

#include "Cancellations.hpp"

#include <torch/torch.h>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

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
     * @param optimizer_name
     * @return
     */
    int mapToOptimizerType(std::string &optimizer_name);

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
     * @param circuit
     * @return
     */
    std::string select_best_agent(const std::string &circuit);


    /**
     *
     * @param agent_name
     * @param circuit
     * @param nr_passes
     * @param output_path
     * @return
     */
    std::tuple<std::vector<std::string>, std::vector<unsigned int> >
    getRecommendedPasses(
        const std::string &agent_name, const std::string &circuit,
        unsigned int nr_passes, fs::path output_path = fs::path());
} // namespace ai_pass_selector

#endif // AGENT_UTILS_HPP
