#include "mlir_utils.hpp"
#include "Environment/environment.hpp"
#include "Torch/A2C/base_a2c_agent.hpp"
#include "Utils/info_utils.hpp"

#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include <Torch/agent_utils.hpp>
#include <Torch/training.hpp>

using namespace ai_pass_selector;
namespace fs = std::filesystem;

void print_help() {
  std::cout <<
      "\nUsage: ./ai_pass_selector_torch [options]\n"
      "Options:\n"
      "  -h, --help               Show this help message\n"
      "  -i, --info               Print info of provided arguments.\n"
      "  -t, --train              Whether to train the agent\n"
      "  -p, --training-params    Depending on the selected agent and training, certain params need to be defined, if not defined, they'll default to their default values.\n"
      "      Training params format: Space seperated list of <param>=<value>\n"
      "  -u, --use                Whether to apply the selected passes onto the circuit.\n"
      "  -c, --circuit <file>     Circuit file (.qasm or .qke). Required if not training\n"
      "  -o, --output <file>      Circuit file path to output the optimized circuit if using the passes.\n"
      "  -d, --dataset <name>     Dataset name (defaults to all)\n"
      "      For available dataset, see AI/Datasets\n"
      "      dataset=all -> Train on all available datasets.\n"
      "  -a, --agent <name>       Agent (defaults to auto)\n"
      "      For available agents, see AI/Agents\n"
      "      agent=auto -> auto-select best agent for circuit dimensions.\n\n";
}

int main(int argc, char **argv) {
  if (argc <= 1) {
    print_help();
    return 1;
  }
  bool info = false;
  bool train = false;
  bool use = false;
  std::string circuit;
  std::string dataset = "all";
  std::string agent_name = "auto";
  std::unordered_map<std::string, std::string> training_params;

  std::vector<std::string> args(argv + 1, argv + argc);
  unsigned int n = args.size();

  unsigned int i = 0;
  while (i < n) {
    if (args[i] == "-h" || args[i] == "--help") {
      print_help();
      return 0;
    }
    if (args[i] == "-i" || args[i] == "--info") {
      info = true;
    } else if (args[i] == "-u" || args[i] == "--use") {
      use = true;
    } else if (args[i] == "-t" || args[i] == "--train") {
      train = true;
    } else if (args[i] == "-p" || args[i] == "--training-params") {
      while (++i < n
             && !args[i].empty()
             && !args[i][0] == '-'
             && args[i].find('=') != std::string::npos) {
        auto pos = args[i].find('=');
        if (pos == std::string::npos || pos == 0 || pos == args[i].size() - 1) {
          std::cerr << "Invalid training param: " << args[i] << std::endl;
          return 1;
        }
        training_params[args[i].substr(0, pos)]
            = args[i].substr(pos + 1);
      }
      // Avoid i++ at the end of the loop
      continue;
    } else if (args[i] == "-c" || args[i] == "--circuit") {
      if (++i < n) {
        circuit = args[i];
      } else {
        std::cerr << "Missing value for --circuit\n";
        return 1;
      }
    } else if (args[i] == "-d" || args[i] == "--dataset") {
      if (++i < n) {
        dataset = args[i];
      } else {
        std::cerr << "Missing value for --dataset\n";
        return 1;
      }
    } else if (args[i] == "-a" || args[i] == "--agent") {
      if (++i < n) {
        agent_name = args[i];
      } else {
        std::cerr << "Missing value for --agent\n";
        return 1;
      }
    } else {
      std::cout << "Unrecognized argument argument: " << args[i] << std::endl;
    }
    i++;
  }

  if (info) {
    if (!circuit.empty()) {
      print_circuit_info(circuit);
    }
    if (!dataset.empty()) {
      print_dataset_info(dataset);
    }
    if (!agent_name.empty()) {
      print_agent_info(agent_name);
    }
    return 0;
  }
  if (agent_name == "auto") {
    if (circuit.empty()) {
      std::cerr << "Cannot select best agent for circuit without circuit.\n";
      return 1;
    }
    agent_name = select_best_agent(circuit);
  }
  if (train) {
    if (dataset.empty()) {
      std::cerr << "Training requires a dataset (--dataset)\n";
      return 1;
    }
    train_agent(agent_name, dataset, training_params);
  }
  if (!circuit.empty()) {
    auto [pass_names, pass_functions]
        = getRecommendedPasses(agent_name, circuit);
    std::cout << "Selected agent: " << agent_name << "\n"
        << "Selected circuit: " << circuit << "\n"
        << "Recommended passes: " << std::endl;
    for (const auto &name : pass_names) {
      std::cout << name << std::endl;
    }
    if (use) {
      if (pass_functions.empty()) {
        std::cerr << "No passes to apply.\n";
        return 1;
      }
      if (!apply_passes_to_circuit(circuit, pass_functions)) {
        std::cerr << "Failed to apply passes to circuit.\n";
        return 1;
      }
    }
  }

  return 0;
}