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

namespace fs = std::filesystem;

void print_help() {
  std::cout <<
      "\nUsage: ./ai_pass_selector_torch [options]\n"
      "Options:\n"
      "  -h, --help               Show this help message\n"
      "  -i, --info               Print info of provided arguments.\n"
      "  -t, --train              Whether to train the agent\n"
      "  -c, --circuit <file>     Circuit file (.qasm or .qke). Required if not training\n"
      "  -d, --dataset <name>     Dataset name (defaults to all)\n"
      "      For available dataset, see AI/Datasets\n"
      "      dataset=all -> Train on all available datasets.\n"
      "  -a, --agent <name>       Agent (defaults to auto)\n"
      "      For available agents, see AI/Agents\n"
      "      agent=auto -> auto-select best agent for circuit dimensions.\n"
      "  -e, --episodes <num>     Number of episodes (defaults to 100000)\n\n";
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
  std::string agent = "auto";
  unsigned long episodes = 100000;

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
        agent = args[i];
      } else {
        std::cerr << "Missing value for --agent\n";
        return 1;
      }
    } else if (args[i] == "-e" || args[i] == "--episodes") {
      if (++i < n) {
        try {
          episodes = std::stoul(args[i]);
        } catch (...) {
          std::cerr << "Invalid value for --episodes\n";
          return 1;
        }
      } else {
        std::cerr << "Missing value for --episodes\n";
        return 1;
      }
    } else {
      std::cout << "Unrecognized argument argument: " << args[i] << std::endl;
    }
    i++;
  }

  if (info) {
    if (!circuit.empty()) {
      ai_pass_selector::print_circuit_info(circuit);
    }
    if (!dataset.empty()) {
      ai_pass_selector::print_dataset_info(dataset);
    }
    if (!agent.empty()) {
      ai_pass_selector::print_agent_info(agent);
    }
    return 0;
  }
  if (agent == "auto") {
    if (circuit.empty()) {
      std::cerr << "Cannot select best agent for circuit without circuit.\n";
      return 1;
    }
    agent = ai_pass_selector::select_best_agent(circuit);
  }
  if (train) {
    if (dataset.empty()) {
      std::cerr << "Training requires a dataset (--dataset)\n";
      return 1;
    }
  }
  if (!circuit.empty()) {
    // TODO:
  }

  return 0;
}