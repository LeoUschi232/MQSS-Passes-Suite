#ifndef AGENT_UTILS_HPP
#define AGENT_UTILS_HPP
#include "NeuralNetworks/Agents/abstract_agent.hpp"


// Passes includes
#include "Cancellations.hpp"

// Torch includes
#include <torch/torch.h>

// Standard Library includes
#include <filesystem>
#include <memory>
#include <unordered_map>

namespace fs = std::filesystem;

using namespace mqss::opt;

namespace ai_pass_selector {
enum class AgentClass : int {
  A3C = 1,
  SAC = 2,
  ACKTR = 3,
  ACER = 4,
  PPO = 5,
  CROSSQ = 6
};

enum class OptimizerType : int {
  Adagrad = 1,
  Adam = 2,
  AdamW = 3,
  LBFGS = 4,
  RMSProp = 5,
  SGD = 6
};

struct EnumClassHash {
  template <typename T> std::size_t operator()(T value) const noexcept {
    return static_cast<std::size_t>(value);
  }
};

struct AgentAttributes {
  AgentClass agent_class;
  unsigned int max_qubits;
  std::string extras;
};

const std::unordered_map<std::string, AgentClass> AGENT_NAME_TO_CLASS = {
    {"a3c", AgentClass::A3C},     {"sac", AgentClass::SAC},
    {"acktr", AgentClass::ACKTR}, {"acer", AgentClass::ACER},
    {"ppo", AgentClass::PPO},     {"crossq", AgentClass::CROSSQ}};

const std::unordered_map<AgentClass, std::string, EnumClassHash>
    AGENT_CLASS_TO_NAME = {
        {AgentClass::A3C, "a3c"},     {AgentClass::SAC, "sac"},
        {AgentClass::ACKTR, "acktr"}, {AgentClass::ACER, "acer"},
        {AgentClass::PPO, "ppo"},     {AgentClass::CROSSQ, "crossq"}};

/// Optimizers
const std::unordered_map<std::string, OptimizerType> OPTIMIZER_NAME_TO_TYPE = {
    {"adagrad", OptimizerType::Adagrad}, {"adam", OptimizerType::Adam},
    {"adamw", OptimizerType::AdamW},     {"lbfgs", OptimizerType::LBFGS},
    {"rmsprop", OptimizerType::RMSProp}, {"sgd", OptimizerType::SGD}};

const std::unordered_map<OptimizerType, std::string, EnumClassHash>
    OPTIMIZER_TYPE_TO_NAME = {
        {OptimizerType::Adagrad, "adagrad"}, {OptimizerType::Adam, "adam"},
        {OptimizerType::AdamW, "adamw"},     {OptimizerType::LBFGS, "lbfgs"},
        {OptimizerType::RMSProp, "rmsprop"}, {OptimizerType::SGD, "sgd"}};

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
makeOptimizer(OptimizerType optimizerType,
              const torch::nn::Sequential &agentModel, double learningRate);

/**
 *
 * @param circuit
 * @return
 */
std::string select_best_agent(const std::string &circuit);

/**
 *
 * @param network
 * @return
 */
unsigned int count_nr_trainable_parameters(const torch::nn::Module &network);
} // namespace ai_pass_selector

#endif // AGENT_UTILS_HPP
