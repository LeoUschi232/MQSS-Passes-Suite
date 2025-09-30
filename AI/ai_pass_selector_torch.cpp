#include "Utils/info_utils.hpp"

// Environemnt includes
#include "Environment/statistics_for_rqcg.hpp"

// Torch includes
#include "Torch/agent_utils.hpp"
#include "Torch/training_and_run_manager.hpp"

// Standard library includes
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

using namespace ai_pass_selector;
namespace fs = std::filesystem;

/// Default values for agent/environment/training parameters.
std::unordered_map<std::string, std::string> load_default_params();

void print_help() {
  std::cout
      << "\nUsage: ./ai_pass_selector_torch [options]\n"
         "Options:\n"
         "  -h, --help                    Show this help message\n"
         "  -a, --agent <name>            Agent to train/use, if not provided, "
         "defaults to most appropriate for circuit.\n"
         "  -d, --dataset <name>          Dataset name, agent will train on "
         "this dataset if provided.\n"
         "  -c, --circuit <file>          Quake circuit file, agent will be "
         "used on this circuit.\n"
         "  -o, --output <file_path>      Circuit file path to output the "
         "optimized circuit if using the passes.\n"
         "  -i, --info                    Print info of provided arguments.\n"
         "Other parameters:\n"
         "  <key>=<value>                 Parameters for "
         "agent/environment/training, will be loaded with default values if "
         "nor provided.\n\n";
}

int main(int argc, char **argv) {
  if (argc <= 1) {
    print_help();
    return 1;
  }
  bool info = false;
  std::unordered_map<std::string, std::string> params = load_default_params();
  std::vector<std::string> args(argv + 1, argv + argc);
  unsigned int n = args.size();

  unsigned int i = 0;
  while (i < n) {
    if (args[i] == "-h" || args[i] == "--help") {
      print_help();
      return 0;
    }
    if (args[i] == "-a" || args[i] == "--agent") {
      if (++i < n) {
        params["agent"] = args[i];
      } else {
        std::cerr << "No agent provided." << std::endl;
        return 1;
      }
    } else if (args[i] == "-d" || args[i] == "--dataset") {
      if (++i < n) {
        params["dataset"] = args[i];
      } else {
        std::cerr << "No dataset provided." << std::endl;
        return 1;
      }
    } else if (args[i] == "-c" || args[i] == "--circuit") {
      if (++i < n) {
        params["circuit"] = args[i];
      } else {
        std::cerr << "No circuit provided." << std::endl;
        return 1;
      }
    } else if (args[i] == "-o" || args[i] == "--output") {
      if (++i < n) {
        params["output"] = args[i];
      } else {
        std::cerr << "No output provided." << std::endl;
        return 1;
      }
    } else if (args[i] == "-i" || args[i] == "--info") {
      info = true;
    } else {
      std::vector<std::string> key_value = split_string(args[i], '=');
      if (key_value.size() != 2) {
        std::cerr << "Malformatted params: " << args[i] << std::endl;
      }
      params[key_value[0]] = key_value[1];
    }
    i++;
  }
  std::string agent = params["agent"];
  std::string dataset = params["dataset"];
  std::string circuit = params["circuit"];
  std::string output = params["output"];

  if (info) {
    std::cout << "Parameters:" << std::endl;
    for (auto [key, value] : params) {
      std::cout << "  " << key << ": " << value << std::endl;
    }
    std::cout << std::endl;
    if (!circuit.empty()) {
      print_circuit_info(circuit);
    }
    if (!dataset.empty()) {
      print_dataset_info(dataset);
      print_dataset_statistics(dataset);
    }
    if (!agent.empty()) {
      print_agent_info(agent);
    }
    return 0;
  }
  if (agent.empty()) {
    if (circuit.empty()) {
      return 0;
    }
    agent = select_best_agent(circuit);
  }
  if (!dataset.empty()) {
    train(agent, dataset, params);
  }
  if (!circuit.empty()) {
    run(agent, circuit, output, params);
  }
  return 0;
}

std::unordered_map<std::string, std::string> load_default_params() {
  return {{"agent", "a2c-mq130-conv2npi"},
          {"dataset", "mqtbench"},
          {"circuit", ""},
          {"output", ""},
          {"nr_parallel_environments", "16"},
          {"episodes", "1000"},
          {"max_steps_per_episode", "5"},
          {"discount_factor", "1.0"},
          {"gae_hyperparameter", "0.96"},
          {"entropy_coefficient", "0.01"},
          {"device", torch::cuda::is_available() ? "cuda" : "cpu"},
          {"critic_optimizer", "adam"},
          {"actor_optimizer", "adam"},
          {"critic_learning_rate", "0.005"},
          {"actor_learning_rate", "0.001"},
          {"print_param_info", "true"}};
}
