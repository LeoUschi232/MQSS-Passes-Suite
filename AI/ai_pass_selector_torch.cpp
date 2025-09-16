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

namespace fs = std::filesystem;

void print_help() {
  std::cout <<
      "\nUsage: ./ai_pass_selector_torch [options]\n"
      "Options:\n"
      "  -h, --help               Show this help message\n"
      "  -i, --info               Print info of provided arguments.\n"
      "  -t, --train              Whether to train the agent\n"
      "  -c, --circuit <file>     Circuit file (.qasm or .qke). Required if not training\n"
      "  -d, --dataset <name>     Dataset name (mandatory if training)\n"
      "      Currently available datasets:\n"
      "      [Filtered, Passtest, Tensortest, Chemistry, Random]\n"
      "  -a, --agent <name>       Agent (mandatory)\n"
      "      Currently available agents:\n"
      "      [a2c]\n"
      "  -e, --episodes <num>     Number of episodes (default 100000, used only when training)\n\n";
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
  std::string dataset;
  std::string agent;
  unsigned long episodes = 100000;

  std::vector<std::string> args(argv + 1, argv + argc);

  for (size_t i = 0; i < args.size(); i++) {
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
      if (i + 1 < args.size()) {
        circuit = args[i + 1];
        i++;
      } else {
        std::cerr << "Missing value for --circuit\n";
        return 1;
      }
    } else if (args[i] == "-d" || args[i] == "--dataset") {
      if (i + 1 < args.size()) {
        dataset = args[i + 1];
        if (!dataset.empty()) {
          dataset[0] = std::toupper(dataset[0]);
          for (size_t j = 1; j < dataset.size(); ++j) {
            dataset[j] = std::tolower(dataset[j]);
          }
        }
        i++;
      } else {
        std::cerr << "Missing value for --dataset\n";
        return 1;
      }
    } else if (args[i] == "-a" || args[i] == "--agent") {
      if (i + 1 < args.size()) {
        agent = args[i + 1];
        i++;
      } else {
        std::cerr << "Missing value for --agent\n";
        return 1;
      }
    } else if (args[i] == "-e" || args[i] == "--episodes") {
      if (i + 1 < args.size()) {
        try {
          episodes = std::stoul(args[i + 1]);
          i++;
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
  }

  std::cout << "train=" << train
      << " agent=" << agent
      << " circuit=" << circuit
      << " dataset=" << dataset
      << " episodes=" << episodes << "\n";
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

  if (!train && circuit.empty()) {
    std::cerr <<
        "Agent not set to train and no circuit to use it on was provided.\n";
    return 1;
  }

  if (train && dataset.empty()) {
    std::cerr << "Training requires a dataset (--dataset)\n";
    return 1;
  }

  std::cout << "train=" << train
      << " agent=" << agent
      << " circuit=" << circuit
      << " dataset=" << dataset
      << " episodes=" << episodes << "\n";

  return 0;
}