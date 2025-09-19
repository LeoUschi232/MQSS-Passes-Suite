#include "Utils/info_utils.hpp"

#include "Quake.hpp"
#include "mlir_utils.hpp"

#include <string>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
int random_int(int start, int end) {
  std::uniform_int_distribution<std::size_t> dist(start, end - 1);
  return dist(rng);
}


std::vector<std::string> split_string(const std::string &str, char delimiter) {
  std::vector<std::string> parts;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    parts.push_back(item);
  }
  return parts;
}

std::optional<fs::path>
search_circuit(const std::string &circuit) {
  auto circuit_path = fs::path(circuit);
  std::string circuit_name = circuit_path.stem().string();

  // First assume the provided circuit is a full filepath to the .qke file.
  if (std::string circuit_extension = circuit_path.extension().string();
    fs::exists(circuit_path) && fs::is_regular_file(circuit_path)
    && circuit_extension == ".qke") {
    return circuit_path;
  }

  auto search_dir = [&](const fs::path &dir) {
    for (auto it = fs::recursive_directory_iterator(
             dir, fs::directory_options::skip_permission_denied);
         it != fs::recursive_directory_iterator(); ++it) {
      if (const fs::directory_entry &entry = *it;
        entry.is_regular_file()) {
        if (const fs::path &entry_path = entry.path();
          entry_path.stem() == circuit_name
          && entry_path.extension() == ".qke") {
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


std::optional<std::tuple<
  fs::path, unsigned int, unsigned int, unsigned int> >
get_circuit_info(const std::string &circuit) {
  auto found_circuit = search_circuit(circuit);
  if (!found_circuit.has_value()) {
    return std::nullopt;
  }
  fs::path circuit_path = found_circuit.value();
  std::string quake_module_text
      = readFileToString(circuit_path.string());
  auto [mlir_module, context_ptr] = extractMLIRContext(quake_module_text);
  auto [nrQubits, nrGates, depth] = getQubitsInstructionsDepth(
      FuncOp(mlir_module));
  return std::make_tuple(circuit_path, nrQubits, nrGates, depth);
}

void print_circuit_info(const std::string &circuit_file) {
  auto circuit_info = get_circuit_info(circuit_file);
  if (!circuit_info.has_value()) {
    std::cerr << "Circuit " + circuit_file + " not found." << std::endl;
    return;
  }
  auto [circuit_path, nr_qubits, nr_gates, depth]
      = circuit_info.value();
  std::string circuit_name = circuit_path.stem().string();
  std::cout << "Circuit " + circuit_name + ":" << std::endl;
  std::cout
      << "   Circuit path: " << circuit_path << "\n"
      << "   Number of qubits: " << nr_qubits << "\n"
      << "   Number of gates: " << nr_gates << "\n"
      << "   Depth: " << depth << std::endl;
}

std::vector<fs::path> get_dataset_files(const std::string &dataset) {
  if (dataset.empty()) {
    return {};
  }
  std::string dataset_name_lower = dataset;
  std::transform(
      dataset_name_lower.begin(),
      dataset_name_lower.end(),
      dataset_name_lower.begin(),
      [](unsigned char c) { return std::tolower(c); });
  std::vector<fs::path> files;
  fs::path quake_dataset_dir;
  if (dataset_name_lower == "all" || dataset == "*") {
    quake_dataset_dir = fs::path(AI_DATASET_DIR) / "Quake";
  } else {
    quake_dataset_dir
        = fs::path(AI_DATASET_DIR) / "Quake" / dataset;
    if (!fs::exists(quake_dataset_dir) || !
        fs::is_directory(quake_dataset_dir)) {
      return {};
    }
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


std::optional<std::tuple<
  unsigned int,
  unsigned int, double, unsigned int,
  unsigned int, double, unsigned int,
  unsigned int, double, unsigned int> >
get_dataset_info(const std::string &dataset_name) {
  if (dataset_name.empty()) {
    return std::nullopt;
  }
  fs::path quake_dataset_dir
      = fs::path(AI_DATASET_DIR) / "Quake" / dataset_name;
  if (!fs::exists(quake_dataset_dir) || !fs::is_directory(quake_dataset_dir)) {
    return std::nullopt;
  }
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
  for (auto entry_path : get_dataset_files(dataset_name)) {
    std::string quake_module_text
        = readFileToString(entry_path.string());
    auto [mlir_module, context_ptr]
        = extractMLIRContext(quake_module_text);
    unsigned int nr_qubits = getNumberOfQubits(FuncOp(mlir_module));
    unsigned int nr_gates = getNumberOfGates(FuncOp(mlir_module));
    unsigned int depth = getCircuitDepth(FuncOp(mlir_module));
    if (nr_qubits < min_nr_qubits) {
      min_nr_qubits = nr_qubits;
    }
    if (nr_qubits > max_nr_qubits) {
      max_nr_qubits = nr_qubits;
    }
    avg_nr_qubits += nr_qubits;
    if (nr_gates < min_nr_gates) {
      min_nr_gates = nr_gates;
    }
    if (nr_gates > max_nr_gates) {
      max_nr_gates = nr_gates;
    }
    avg_nr_gates += nr_gates;
    if (depth < min_depth) {
      min_depth = depth;
    }
    if (depth > max_depth) {
      max_depth = depth;
    }
    avg_depth += depth;
    nr_circuits++;
  }
  if (nr_circuits <= 0) {
    return std::nullopt;
  }
  avg_nr_qubits /= nr_circuits;
  avg_nr_gates /= nr_circuits;
  avg_depth /= nr_circuits;
  return std::make_tuple(
      nr_circuits,
      min_nr_qubits, avg_nr_qubits, max_nr_qubits,
      min_nr_gates, avg_nr_gates, max_nr_gates,
      min_depth, avg_depth, max_depth);
}


void print_dataset_info(const std::string &dataset_name) {
  if (dataset_name.empty()) {
    return;
  }
  fs::path quake_dataset_dir =
      fs::path(AI_DATASET_DIR) / "Quake" / dataset_name;

  if (fs::exists(quake_dataset_dir) && fs::is_directory(quake_dataset_dir)) {
    std::cout << "Dataset Quake/" + dataset_name << ":" << std::endl;
    auto dataset_info = get_dataset_info(dataset_name);
    if (!dataset_info.has_value()) {
      std::cout << "   No circuits found." << std::endl;
      return;
    }
    auto [nr_circuits,
      min_nr_qubits, avg_nr_qubits, max_nr_qubits,
      min_nr_gates, avg_nr_gates, max_nr_gates,
      min_depth, avg_depth, max_depth] = dataset_info.value();
    if (nr_circuits <= 0) {
      std::cout << "   No circuits found." << std::endl;
      return;
    }
    std::cout
        << "   Number of circuits: " << nr_circuits << "\n"
        << "   Minimum number of qubits: " << min_nr_qubits << "\n"
        << "   Average number of qubits: " << avg_nr_qubits << "\n"
        << "   Maximum number of qubits: " << max_nr_qubits << "\n"
        << "   Minimum number of gates: " << min_nr_gates << "\n"
        << "   Average number of gates: " << avg_nr_gates << "\n"
        << "   Maximum number of gates: " << max_nr_gates << "\n"
        << "   Minimum depth: " << min_depth << "\n"
        << "   Average depth: " << avg_depth << "\n"
        << "   Maximum depth: " << max_depth << std::endl;
    return;
  }
  std::cout << "Dataset Quake/" + dataset_name << " not found." << std::endl;
  fs::path qasm_dataset_dir = fs::path(AI_DATASET_DIR)
                              / "Qasm" / dataset_name;
  if (fs::exists(qasm_dataset_dir) && fs::is_directory(qasm_dataset_dir)) {
    unsigned int nr_circuits = 0;
    for (auto it = fs::recursive_directory_iterator(
             qasm_dataset_dir, fs::directory_options::skip_permission_denied);
         it != fs::recursive_directory_iterator(); ++it) {
      if (const fs::directory_entry &entry = *it; entry.is_regular_file()) {
        if (entry.path().extension() == ".qasm") {
          nr_circuits++;
        }
      }
    }
    std::cout << "Dataset Qasm/" + dataset_name << " found with "
        << nr_circuits << " circuits.\n"
        << "Run \"./convert_qasm_dataset_to_quake " << dataset_name
        << "\" to convert this dataset to Quake." << std::endl;
    return;
  }
  std::cout << "Dataset Qasm/" + dataset_name + " not found." << std::endl;
}

void print_agent_info(const std::string &agent_name) {
  throw std::runtime_error("Not implemented yet");
}

} // namespace ai_pass_selector