#ifndef AGENT_UTILS_HPP
#define AGENT_UTILS_HPP

#include "Cancellations.hpp"

#include <torch/torch.h>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

using namespace mqss::opt;

namespace ai_pass_selector {
    /// Agent classes
    constexpr int A2C = 1;
    constexpr int A3C = 2;
    constexpr int PPO = 3;
    constexpr int RNN = 4;

    const std::unordered_map<std::string, int> AGENT_NAME_TO_CLASS = {
        {"a2c", A2C},
        {"a3c", A3C},
        {"ppo", PPO},
        {"rnn", RNN}
    };

    /// Optimizers
    constexpr int OPTIMIZER_ADAGRAD = 1;
    constexpr int OPTIMIZER_ADAM = 2;
    constexpr int OPTIMIZER_ADAMW = 3;
    constexpr int OPTIMIZER_LBFGS = 4;
    constexpr int OPTIMIZER_RMSPROP = 5;
    constexpr int OPTIMIZER_SGD = 6;

    const std::unordered_map<std::string, int> OPTIMIZER_NAME_TO_TYPE = {
        {"adagrad", OPTIMIZER_ADAGRAD},
        {"adam", OPTIMIZER_ADAM},
        {"adamw", OPTIMIZER_ADAMW},
        {"lbfgs", OPTIMIZER_LBFGS},
        {"rmsprop", OPTIMIZER_RMSPROP},
        {"sgd", OPTIMIZER_SGD}
    };

    struct AgentAttributes {
        int agent_class;
        int size_class;
        std::string specifier;
    };

    /// Devices
    const std::unordered_map<std::string, torch::Device> DEVICE_NAME_TO_TORCH = {
        {"cpu", torch::kCPU},
        {"cuda", torch::kCUDA},
        {"gpu", torch::kCUDA}
    };

    /**
     *
     * @param agent_name
     * @return
     */
    AgentAttributes parseAgentName(const std::string &agent_name);

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
