#ifndef AGENT_UTILS_HPP
#define AGENT_UTILS_HPP

// Passes includes
#include "Cancellations.hpp"

// Torch includes
#include <torch/torch.h>

// Standard Library includes
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

using namespace mqss::opt;

namespace ai_pass_selector {
struct AgentAttributes {
  int agent_class;
  unsigned int max_qubits;
  std::string extras;
};

/// Agent classes
constexpr int A3C = 1;
constexpr int SAC = 2;
constexpr int ACKTR = 3;
constexpr int ACER = 4;
constexpr int PPO = 5;
constexpr int CROSSQ = 6;

const std::unordered_map<std::string, int> AGENT_NAME_TO_CLASS = {
    {"a3c", A3C},   {"sac", SAC}, {"acktr", ACKTR},
    {"acer", ACER}, {"ppo", PPO}, {"crossq", CROSSQ}};

const std::unordered_map<int, std::string> AGENT_CLASS_TO_NAME = {
    {A3C, "a3c"},   {SAC, "sac"}, {ACKTR, "acktr"},
    {ACER, "acer"}, {PPO, "ppo"}, {CROSSQ, "crossq"}};

/// Optimizers
constexpr int OPTIMIZER_ADAGRAD = 1;
constexpr int OPTIMIZER_ADAM = 2;
constexpr int OPTIMIZER_ADAMW = 3;
constexpr int OPTIMIZER_LBFGS = 4;
constexpr int OPTIMIZER_RMSPROP = 5;
constexpr int OPTIMIZER_SGD = 6;

const std::unordered_map<std::string, int> OPTIMIZER_NAME_TO_TYPE = {
    {"adagrad", OPTIMIZER_ADAGRAD}, {"adam", OPTIMIZER_ADAM},
    {"adamw", OPTIMIZER_ADAMW},     {"lbfgs", OPTIMIZER_LBFGS},
    {"rmsprop", OPTIMIZER_RMSPROP}, {"sgd", OPTIMIZER_SGD}};

const std::unordered_map<int, std::string> OPTIMIZER_TYPE_TO_NAME = {
    {OPTIMIZER_ADAGRAD, "adagrad"}, {OPTIMIZER_ADAM, "adam"},
    {OPTIMIZER_ADAMW, "adamw"},     {OPTIMIZER_LBFGS, "lbfgs"},
    {OPTIMIZER_RMSPROP, "rmsprop"}, {OPTIMIZER_SGD, "sgd"}};

/// Devices
const std::unordered_map<std::string, torch::Device> DEVICE_NAME_TO_TORCH = {
    {"cpu", torch::kCPU}, {"cuda", torch::kCUDA}, {"gpu", torch::kCUDA}};

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
std::unique_ptr<torch::optim::Optimizer>
makeOptimizer(int optimizerType, const torch::nn::Sequential &agentModel,
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
std::tuple<std::vector<std::string>, std::vector<unsigned int>>
getRecommendedPasses(const std::string &agent_name, const std::string &circuit,
                     unsigned int nr_passes, fs::path output_path = fs::path());
} // namespace ai_pass_selector

#endif // AGENT_UTILS_HPP
