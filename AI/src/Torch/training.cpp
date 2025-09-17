#include "Torch/training.hpp"

// Environment includes
#include <Environment/environment.hpp>

// Torch includes
#include <Torch/A2C/base_a2c_agent.hpp>

// Utils includes
#include <Utils/passes_utils.hpp>
#include <Utils/circuit_utils.hpp>

// Stdandard library includes
#include <unordered_map>
#include <string>
#include <Torch/A2C/a2c_ib_fc_lsd.hpp>
#include <Torch/A2C/a2c_ib_fc_lsm.hpp>

namespace fs = std::filesystem;

namespace ai_pass_selector {
std::unordered_map<std::string, std::string> train_agent(
    const std::string &agent_name,
    const std::string &dataset,
    std::unordered_map<std::string, std::string> training_params) {
  std::unordered_map<std::string, std::string> training_results;
  std::vector<std::string> agent_attributes = split_string(agent_name, '-');

  if (agent_attributes[0] == "a2c") {
    std::vector<std::string> dimensions
        = split_string(agent_attributes[4], 'x');
    unsigned int max_qubits = std::stoi(dimensions[0]);
    unsigned int max_instructions = std::stoi(dimensions[1]);
    unsigned int max_depth = std::stoi(dimensions[2]);

    // Defaults for A2C
    int critic_optimizer_type
        = training_params.find("critic_optimizer") != training_params.end()
            ? mapToOptimizerType(training_params["critic_optimizer"])
            : OPTIMIZER_ADAM;
    int actor_optimizer_type
        = training_params.find("actor_optimizer") != training_params.end()
            ? mapToOptimizerType(training_params["actor_optimizer"])
            : OPTIMIZER_ADAM;
    double critic_learning_rate
        = training_params.find("critic_learning_rate") != training_params.end()
            ? std::stod(training_params["critic_learning_rate"])
            : 0.005;
    double actor_learning_rate
        = training_params.find("actor_learning_rate") != training_params.end()
            ? std::stod(training_params["actor_learning_rate"])
            : 0.001;
    unsigned int nr_parallel_environments
        = training_params.find("nr_parallel_environments") != training_params.
          end()
            ? std::stoi(training_params["nr_parallel_environments"])
            : 10;
    torch::Device device
        = training_params.find("device") != training_params.end()
          && training_params["device"] == "cuda"
            ? torch::kCUDA
            : torch::kCPU;
    unsigned int episodes
        = training_params.find("episodes") != training_params.end()
            ? std::stoi(training_params["episodes"])
            : 100000;
    double discount_factor
        = training_params.find("discount_factor") != training_params.end()
            ? std::stod(training_params["discount_factor"])
            : 1.0;
    double gae_hyperparameter
        = training_params.find("gae_hyperparameter") != training_params.end()
            ? std::stod(training_params["gae_hyperparameter"])
            : 0.96;
    double entropy_coefficient
        = training_params.find("entropy_coefficient") != training_params.end()
            ? std::stod(training_params["entropy_coefficient"])
            : 0.01;
    unsigned int max_steps_per_episode
        = training_params.find("max_steps_per_episode") != training_params.end()
            ? std::stoi(training_params["max_steps_per_episode"])
            : 20;

    if (agent_attributes[1] == "ib") {
      if (agent_attributes[2] == "fc") {
        if (agent_attributes[3] == "lsd") {
          A2C_IB_FC_LSD agent(
              max_qubits, max_instructions, max_depth,
              critic_optimizer_type, actor_optimizer_type,
              critic_learning_rate, actor_learning_rate,
              nr_parallel_environments, device);
          training_results = train_agent(
              agent, dataset, episodes,
              discount_factor, gae_hyperparameter, entropy_coefficient,
              max_steps_per_episode);
        } else if (agent_attributes[3] == "lsm") {
          A2C_IB_FC_LSM agent(
              max_qubits, max_instructions, max_depth,
              critic_optimizer_type, actor_optimizer_type,
              critic_learning_rate, actor_learning_rate,
              nr_parallel_environments, device);
          training_results = train_agent(
              agent, dataset, episodes,
              discount_factor, gae_hyperparameter, entropy_coefficient,
              max_steps_per_episode);
        }
      }
    }

  }
  return training_results;
}

} // namespace ai_pass_selector