#include "Utils/info_utils.hpp"

// Environemnt includes
#include "Environment/statistics_for_rqcg.hpp"

// Torch includes
#include <torch/torch.h>

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"
#include "NeuralNetworks/Agents/training_and_run_manager.hpp"

// Standard library includes
#include <iostream>
#include <string>
#include <vector>

using namespace ai_pass_selector;
namespace fs = std::filesystem;

/// Default values for agent/environment/training parameters.
/// ssh -Y ge78zic2@cool.hpc.lrz.de
/// 7McMGcmhX_27McMGcmhX_2
void load_default_params();

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
  bool info = false;
  load_default_params();
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
        GLOBAL_PARAMS["agent"] = args[i];
      } else {
        std::cerr << "No agent provided." << std::endl;
        return 1;
      }
    } else if (args[i] == "-d" || args[i] == "--dataset") {
      if (++i < n) {
        GLOBAL_PARAMS["dataset"] = args[i];
      } else {
        std::cerr << "No dataset provided." << std::endl;
        return 1;
      }
    } else if (args[i] == "-c" || args[i] == "--circuit") {
      if (++i < n) {
        GLOBAL_PARAMS["circuit"] = args[i];
      } else {
        std::cerr << "No circuit provided." << std::endl;
        return 1;
      }
    } else if (args[i] == "-o" || args[i] == "--output") {
      if (++i < n) {
        GLOBAL_PARAMS["output"] = args[i];
      } else {
        std::cerr << "No output provided." << std::endl;
        return 1;
      }
    } else if (args[i] == "-i" || args[i] == "--info") {
      info = true;
    } else {
      std::vector<std::string> key_value = split_string(args[i], '=');
      if (key_value.size() != 2) {
        std::cerr << "Malformatted GLOBAL_PARAMS: " << args[i] << std::endl;
      }
      GLOBAL_PARAMS[key_value[0]] = key_value[1];
    }
    i++;
  }
  auto agent = std::string(GLOBAL_PARAMS["agent"]);
  auto dataset = std::string(GLOBAL_PARAMS["dataset"]);
  auto circuit = std::string(GLOBAL_PARAMS["circuit"]);
  auto output = std::string(GLOBAL_PARAMS["output"]);

  if (info) {
    std::cout << "Parameters:" << std::endl;
    for (auto [key, value] : GLOBAL_PARAMS) {
      std::cout << "  " << key << ": " << value.to_string() << std::endl;
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
    train(agent, dataset);
  }
  if (!circuit.empty()) {
    run(agent, circuit, output);
  }
  return 0;
}

void load_default_params() {
  GLOBAL_PARAMS = {
      {"agent", "a3c-mq130-tcnrelu"},
      {"dataset", "mqtbench"},
      {"circuit", ""},
      {"output", ""},
      {"nr_asynchronous_agents", 1},
      {"a3c_max_async_steps", 100000},
      {"nr_episodes", 100},
      {"max_steps_per_episode", 130},
      {"max_steps_no_improvement", 13},
      {"max_steps_no_change", 6},
      {"max_steps_same_action", 3},
      {"discount_factor", 0.995},
      {"gae_hyperparameter", 0.96},
      {"entropy_coefficient", 0.01},
      {"device", torch::cuda::is_available() ? torch::kCUDA : torch::kCPU},
      {"critic_optimizer_idx", static_cast<int>(OptimizerType::Adam)},
      {"actor_optimizer_idx", static_cast<int>(OptimizerType::Adam)},
      {"actor_learning_rate", 0.001},
      {"critic_learning_rate", 0.005},
      {"ppo_epsilon", 0.2},
      {"sac_alpha", 0.1},
      {"print_param_info", false},
      {"save_agent_after_training", false},
      {"stop_training_on_error", true},
      {"print_diagnostics", false}};
}
