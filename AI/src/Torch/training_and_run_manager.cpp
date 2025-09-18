#include "Torch/training_and_run_manager.hpp"

// Environment includes
#include <Environment/environment.hpp>

// Torch includes
#include <Torch/A2C/base_a2c_agent.hpp>

// Utils includes
#include <Utils/passes_utils.hpp>

// Stdandard library includes
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <Torch/A2C/a2c_ib_fc_lsd.hpp>
#include <Torch/A2C/a2c_ib_fc_lsm.hpp>
#include <Torch/A2C/a2c_trainer.hpp>

namespace fs = std::filesystem;

namespace ai_pass_selector {
std::unordered_map<std::string, std::string> train(
    const std::string &agent_name,
    const std::string &dataset,
    std::unordered_map<std::string, std::string> params) {
  std::unordered_map<std::string, std::string> training_results;

  switch (AgentAttributes attributes = parseAgentName(agent_name);
    attributes.agent_class) {
  case A2C:
    if (attributes.specific_attributes.size() < 3) {
      throw std::runtime_error(
          "Invalid agent name '" + agent_name
          + "'. Expected format 'a2c-<size>-ib-fc-<lsd|lsm>'.");
    }
    if (attributes.specific_attributes[0] == "ib") {
      if (attributes.specific_attributes[1] == "fc") {
        if (attributes.specific_attributes[2] == "lsd") {
          try {
            A2C_IB_FC_LSD agent(
                attributes.size_class, std::move(params));
            training_results = train_a2c(agent, dataset, std::move(params));
          } catch (const std::runtime_error &e) {
            std::cerr << e.what() << std::endl;
            return {};
          }
        } else if (attributes.specific_attributes[2] == "lsm") {
          try {
            A2C_IB_FC_LSM agent(
                attributes.size_class, std::move(params));
            training_results = train_a2c(agent, dataset, std::move(params));
          } catch (const std::runtime_error &e) {
            std::cerr << e.what() << std::endl;
            return {};
          }
        }
      }
    }
    break;
  case A3C:
    break;
  case PPO:
    break;
  case RNN:
    break;
  default:
    break;
  }
  return training_results;
}

std::unordered_map<std::string, std::string> run(
    const std::string &agent_name,
    const std::string &circuit,
    const std::string &output,
    std::unordered_map<std::string, std::string> params) {
  throw std::runtime_error("Not implemented yet");
}

} // namespace ai_pass_selector