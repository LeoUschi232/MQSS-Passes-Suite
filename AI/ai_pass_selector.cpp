#include "Utils/info_utils.hpp"

// Environemnt includes
#include "Environment/statistics_for_rqcg.hpp"

// Torch includes
#include <torch/torch.h>

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"
#include "NeuralNetworks/Agents/training_and_run_manager.hpp"

// Standard library includes
#include <csignal>
#include <iostream>
#include <string>
#include <vector>

using namespace ai_pass_selector;
namespace fs = std::filesystem;

/// Default values for agent/environment/training parameters.
void load_default_params();

/// Global flag for SIGINT Ctrl+C interruptions.
volatile sig_atomic_t interrupted = 0;
void signal_handler(int signal) {
  if (signal == SIGINT) {
    interrupted = 1;
  }
}

void print_help() {
  std::cout
      << "\nUsage: ./ai_pass_selector_torch [options]\n"
         "Options:\n"
         "  -h, --help                    Show this help message\n"
         "  -a, --agent <name>            Agent to train/use, if not provided, "
         "defaults to most appropriate for circuit.\n"
         "  -d, --dataset <name>          Dataset name, agent will train on "
         "this dataset if provided.\n"
         "-e, --evaluate                  Evaluate the agent instead of "
         "training it.\n"
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
    } else if (args[i] == "-e" || args[i] == "--evaluate") {
      GLOBAL_PARAMS["evaluate"] = true;
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
  GLOBAL_TENSOR_OPTIONS = torch::TensorOptions()
                              .device(GLOBAL_PARAMS["device"].to_device_type())
                              .dtype(torch::kFloat32);

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

  // If a circuit is provided, it is assumes the user only wants to run the
  // passes selection on the circuit and not train the agent so by default do
  // not train if an input circuit is provided.
  if (!dataset.empty() && circuit.empty()) {
    if (GLOBAL_PARAMS["evaluate"].to_bool()) {
      std::optional<unsigned int> max_circuits = std::nullopt;
      if (GLOBAL_PARAMS.find("evaluation_sample") != GLOBAL_PARAMS.end() &&
          GLOBAL_PARAMS["evaluation_sample"].to_int() > 0) {
        max_circuits = GLOBAL_PARAMS["evaluation_sample"].to_int();
      }

      std::unordered_map<std::string, std::string> metrics =
          evaluate(agent, dataset, max_circuits);
      for (const auto &[key, value] : metrics) {
        std::cout << key << ": " << value << std::endl;
      }
    } else {
      train(agent, dataset);
    }
  }
  if (!circuit.empty()) {
    run(agent, circuit, output);
  }
  return 0;
}

void load_default_params() {
  GLOBAL_PARAMS = {
      {"agent", "a2c-mq28-tcn"},
      {"dataset", "Chemistry"},
      {"evaluate", false},
      {"circuit", ""},
      {"output", ""},
      {"nr_episodes", 1000000},
      {"max_steps_per_episode", 256},
      {"max_steps_no_change", 32},
      {"max_steps_same_action", 8},
      {"discount_factor", 0.995},
      {"gae_hyperparameter", 0.96},
      {"entropy_coefficient", 0.01},
      {"device", torch::cuda::is_available() ? torch::kCUDA : torch::kCPU},
      {"critic_optimizer_idx", static_cast<int>(OptimizerType::Adam)},
      {"actor_optimizer_idx", static_cast<int>(OptimizerType::Adam)},
      {"actor_learning_rate", 1e-3},
      {"critic_learning_rate", 5e-3},
      {"sdsac_shared_learning_rate", 3e-4},
      {"ppo_epsilon", 0.2},
      {"sdsac_temperature_alpha", 0.1},
      {"sdsac_smoothing_tau", 0.005},
      {"sdsac_penasdlty_beta", 0.5},
      {"sdsac_clip_c", 0.5},
      {"sdsac_entropy_target_weight", 0.98},
      {"acer_max_nr_trajectories", 500},
      {"acer_ratio_of_replay", 8},
      {"acer_truncation_threshold_c", 10.0},
      {"acer_soft_update_alpha", 0.99},
      {"acer_trust_region_delta", 1.0},
      {"print_param_info", false},
      {"save_agent_after_training", true},
      {"save_agent_every_ith_episode", 10},
      {"stop_training_on_error", true},
      {"print_diagnostics", false},
      {"probability_max_qubits", 0.4},
      {"nr_gates_reduction_weight", 0.15},
      {"evaluation_sample", 10}};
}
