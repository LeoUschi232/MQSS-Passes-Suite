#ifndef AGENT_UTILS_HPP
#define AGENT_UTILS_HPP

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
  PPO = 2,
  SDSAC = 3,
  ACER = 4,
  CROSSQ = 5
};

enum class OptimizerType : int {
  Adagrad = 1,
  Adam = 2,
  AdamW = 3,
  RMSProp = 4,
  SGD = 5
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
    {"a3c", AgentClass::A3C},
    {"ppo", AgentClass::PPO},
    {"acer", AgentClass::ACER},
    {"sdsac", AgentClass::SDSAC},
    {"crossq", AgentClass::CROSSQ}};

const std::unordered_map<AgentClass, std::string, EnumClassHash>
    AGENT_CLASS_TO_NAME = {{AgentClass::A3C, "a3c"},
                           {AgentClass::PPO, "ppo"},
                           {AgentClass::ACER, "acer"},
                           {AgentClass::SDSAC, "sdsac"},
                           {AgentClass::CROSSQ, "crossq"}};

/// Optimizers
const std::unordered_map<std::string, OptimizerType> OPTIMIZER_NAME_TO_TYPE = {
    {"adagrad", OptimizerType::Adagrad},
    {"adam", OptimizerType::Adam},
    {"adamw", OptimizerType::AdamW},
    {"rmsprop", OptimizerType::RMSProp},
    {"sgd", OptimizerType::SGD}};

const std::unordered_map<OptimizerType, std::string, EnumClassHash>
    OPTIMIZER_TYPE_TO_NAME = {{OptimizerType::Adagrad, "adagrad"},
                              {OptimizerType::Adam, "adam"},
                              {OptimizerType::AdamW, "adamw"},
                              {OptimizerType::RMSProp, "rmsprop"},
                              {OptimizerType::SGD, "sgd"}};

/**
 * @param agent_name
 * @return
 */
AgentAttributes parseAgentName(const std::string &agent_name);

/**
 * @param optimizerType
 * @param agentModel
 * @param learningRate
 * @return
 */
std::unique_ptr<torch::optim::Optimizer>
makeOptimizer(OptimizerType optimizerType,
              const torch::nn::Sequential &agentModel, double learningRate);

/**
 * @param circuit
 * @return
 */
std::string select_best_agent(const std::string &circuit);

/**
 * @param network
 * @return
 */
unsigned int count_nr_trainable_parameters(const torch::nn::Module &network);
} // namespace ai_pass_selector

#endif // AGENT_UTILS_HPP
