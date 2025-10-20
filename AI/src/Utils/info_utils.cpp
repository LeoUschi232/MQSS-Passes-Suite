#include "Utils/info_utils.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/A3C/a3c_agents.hpp"
#include "NeuralNetworks/Agents/A3C/base_a3c_agent.hpp"
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Support includes
#include "Support/mlir_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
/// The usages of llvm functions must come before the QuakeOps header which
/// expects them.
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
////////////////////////////////////////////////////////////////////////////////

// Cudaq includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

// Utils includes
#include "Utils/progress_bar.hpp"

// Standard library includes
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
bool isclose(double a, double b, double atol) { return std::abs(a - b) < atol; }

std::vector<std::string> split_string(const std::string &str, char delimiter) {
  std::vector<std::string> parts;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    parts.push_back(item);
  }
  return parts;
}

std::optional<fs::path> search_circuit(const std::string &circuit) {
  auto circuit_path = fs::path(circuit);
  std::string circuit_name = circuit_path.stem().string();

  // First assume the provided circuit is a full filepath to the .qke file.
  if (std::string circuit_extension = circuit_path.extension().string();
      fs::exists(circuit_path) && fs::is_regular_file(circuit_path) &&
      circuit_extension == ".qke") {
    return circuit_path;
  }

  auto search_dir = [&](const fs::path &dir) {
    for (auto it = fs::recursive_directory_iterator(
             dir, fs::directory_options::skip_permission_denied);
         it != fs::recursive_directory_iterator(); ++it) {
      if (const fs::directory_entry &entry = *it; entry.is_regular_file()) {
        if (const fs::path &entry_path = entry.path();
            entry_path.stem() == circuit_name &&
            entry_path.extension() == ".qke") {
          return entry_path;
        }
      }
    }
    return fs::path();
  };

  // Try to find the circuit in current working directory
  fs::path found_path = search_dir(fs::current_path());
  if (!found_path.empty()) {
    return found_path;
  }
  // Try to find the circuit in circuit directory
  found_path = search_dir(fs::path(AI_CIRCUITS_DIR));
  if (!found_path.empty()) {
    return found_path;
  }
  return std::nullopt;
}

std::optional<std::tuple<fs::path, unsigned int, unsigned int, unsigned int>>
get_circuit_info(const std::string &circuit) {
  auto found_circuit = search_circuit(circuit);
  if (!found_circuit.has_value()) {
    return std::nullopt;
  }
  fs::path circuit_path = found_circuit.value();
  std::string quake_module_text = readFileToString(circuit_path.string());
  auto [mlir_module, context_ptr] = extractMLIRContext(quake_module_text);
  auto [nrQubits, nrGates, depth] =
      getQubitsInstructionsDepth(FuncOp(mlir_module));
  return std::make_tuple(circuit_path, nrQubits, nrGates, depth);
}

void print_circuit_info(const std::string &circuit_file) {
  auto circuit_info = get_circuit_info(circuit_file);
  if (!circuit_info.has_value()) {
    std::cerr << "Circuit " + circuit_file + " not found." << std::endl;
    return;
  }
  auto [circuit_path, nr_qubits, nr_gates, depth] = circuit_info.value();
  std::string circuit_name = circuit_path.stem().string();
  std::cout << "Circuit " + circuit_name + ":" << std::endl;
  std::cout << "   Circuit path: " << circuit_path << "\n"
            << "   Number of qubits: " << nr_qubits << "\n"
            << "   Number of gates: " << nr_gates << "\n"
            << "   Depth: " << depth << std::endl;
}

std::vector<fs::path> get_dataset_files(const std::string &dataset_name) {
  if (dataset_name.empty()) {
    return {};
  }
  std::vector<fs::path> files;
  fs::path quake_dataset_dir;
  if (dataset_name == "all" || dataset_name == "*") {
    quake_dataset_dir = fs::path(AI_DATASET_DIR) / "Quake";
  } else {
    quake_dataset_dir = fs::path(AI_DATASET_DIR) / "Quake" / dataset_name;
  }
  if (!fs::exists(quake_dataset_dir) || !fs::is_directory(quake_dataset_dir)) {
    return {};
  }
  for (auto it = fs::recursive_directory_iterator(
           quake_dataset_dir, fs::directory_options::skip_permission_denied);
       it != fs::recursive_directory_iterator(); ++it) {
    const fs::directory_entry &entry = *it;
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path &entry_path = entry.path();
    if (entry_path.extension() != ".qke") {
      continue;
    }
    files.push_back(entry_path);
  }
  return files;
}

std::optional<std::vector<std::pair<std::string, std::string>>>
get_dataset_info(const std::string &dataset_name) {
  std::vector<fs::path> files = get_dataset_files(dataset_name);
  if (files.empty()) {
    return std::nullopt;
  }
  unsigned int nr_files = files.size();
  unsigned int max_int = std::numeric_limits<unsigned int>::max();
  unsigned int nr_circuits = 0;
  unsigned int min_nr_qubits = max_int;
  double avg_nr_qubits = 0;
  unsigned int max_nr_qubits = 0;
  unsigned int min_nr_gates = max_int;
  double avg_nr_gates = 0;
  unsigned int max_nr_gates = 0;
  unsigned int min_depth = max_int;
  double avg_depth = 0;
  unsigned int max_depth = 0;

  unsigned int progress = 0;
  for (auto entry_path : files) {
    updateProgress(++progress, nr_files, "Retrieving info of: " + dataset_name);
    std::string quake_module_text = readFileToString(entry_path.string());
    auto [circuit, context_ptr] = extractMLIRContext(quake_module_text);
    auto [nr_qubits, nr_gates, depth] =
        getQubitsInstructionsDepth(FuncOp(circuit));
    min_nr_qubits = std::min(min_nr_qubits, nr_qubits);
    avg_nr_qubits += nr_qubits;
    max_nr_qubits = std::max(max_nr_qubits, nr_qubits);
    min_nr_gates = std::min(min_nr_gates, nr_gates);
    avg_nr_gates += nr_gates;
    max_nr_gates = std::max(max_nr_gates, nr_gates);
    min_depth = std::min(min_depth, depth);
    avg_depth += depth;
    max_depth = std::max(max_depth, depth);
    nr_circuits++;
  }
  std::cout << std::endl;
  if (nr_circuits <= 0) {
    return std::nullopt;
  }
  avg_nr_qubits /= nr_circuits;
  avg_nr_gates /= nr_circuits;
  avg_depth /= nr_circuits;
  std::cout << std::endl;
  std::vector<std::pair<std::string, std::string>> info;
  info.emplace_back("Dataset name", dataset_name);
  info.emplace_back("Number of circuits", std::to_string(nr_circuits));
  info.emplace_back("Minimum number of qubits", std::to_string(min_nr_qubits));
  info.emplace_back("Average number of qubits", std::to_string(avg_nr_qubits));
  info.emplace_back("Maximum number of qubits", std::to_string(max_nr_qubits));
  info.emplace_back("Minimum number of gates", std::to_string(min_nr_gates));
  info.emplace_back("Average number of gates", std::to_string(avg_nr_gates));
  info.emplace_back("Maximum number of gates", std::to_string(max_nr_gates));
  info.emplace_back("Minimum depth", std::to_string(min_depth));
  info.emplace_back("Average depth", std::to_string(avg_depth));
  info.emplace_back("Maximum depth", std::to_string(max_depth));
  return info;
}

void print_dataset_info(const std::string &dataset_name) {
  std::vector<std::pair<std::string, std::string>> info =
      get_dataset_info(dataset_name)
          .value_or(std::vector<std::pair<std::string, std::string>>{});
  if (info.empty()) {
    std::cerr << "Dataset " + dataset_name + " not found or empty."
              << std::endl;
    return;
  }
  for (const auto &[key, value] : info) {
    std::cout << key + ": " + value << std::endl;
  }
  std::cout << std::endl;
}

void print_agent_info(const std::string &agent_name) {
  unsigned int nr_parameters = 0;
  try {
    switch (AgentAttributes attributes = parseAgentName(agent_name);
            attributes.agent_class) {
    case AgentClass::A3C: {
      std::unique_ptr<BaseA3CAgent> agent;
      if (attributes.extras == "tcnrelu") {
        agent = std::make_unique<A3C_TCN_RELU>(attributes.max_qubits);
      } else if (attributes.extras == "tcnprelu") {
        agent = std::make_unique<A3C_TCN_PRELU>(attributes.max_qubits);
      } else if (attributes.extras == "lstmhmpp") {
        agent = std::make_unique<A3C_LSTM_HMPP>(attributes.max_qubits);
      } else if (attributes.extras == "lstmbmnp") {
        agent = std::make_unique<A3C_LSTM_BMNP>(attributes.max_qubits);
      } else if (attributes.extras == "hybrid") {
        agent = std::make_unique<A3C_HYBRID>(attributes.max_qubits);
      } else {
        std::cerr << "No such A3C agent: " << agent_name << std::endl;
        return;
      }
      nr_parameters = nr_trainable_parameters(*agent);
      break;
    }
    default:
      std::cerr << "No such agent yet: " << agent_name << std::endl;
      return;
    }
  } catch (const std::runtime_error &e) {
    std::cerr << "\n" << e.what() << std::endl;
    return;
  }
  std::cout << "\nAgent name: " << agent_name << "\n"
            << "Agent nr trainable parameters: " << nr_parameters << std::endl;
}

} // namespace ai_pass_selector