#include "Utils/info_utils.hpp"

#include "mlir_utils.hpp"

#include <string>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
std::optional<std::tuple<fs::path, std::string, std::string> >
search_circuit(const std::string &circuit_file) {
  fs::path circuit_path = circuit_file;
  std::string circuit_name = circuit_path.stem().string();
  std::string circuit_extension = circuit_path.extension().string();
  fs::path circuit_folder;

  if (fs::exists(circuit_path) && fs::is_regular_file(circuit_path)
      && (circuit_extension == ".qke" || circuit_extension == ".qasm")) {
    return std::make_tuple(
        fs::absolute(circuit_path).parent_path(),
        circuit_name, circuit_extension);
  }
  fs::path dataset_dir = fs::path(AI_DATASET_DIR) / "Quake";
  for (auto it = fs::recursive_directory_iterator(
           dataset_dir, fs::directory_options::skip_permission_denied);
       it != fs::recursive_directory_iterator(); ++it) {
    if (const fs::directory_entry &entry = *it;
      entry.is_regular_file()) {
      if (const fs::path &entry_path = entry.path();
        entry_path.stem() == circuit_name
        && entry_path.extension() == ".qke") {
        return std::make_tuple(
            fs::absolute(entry_path).parent_path(), circuit_name, ".qke");
      }
    }
  }
  dataset_dir = fs::path(AI_DATASET_DIR) / "Qasm";
  for (auto it = fs::recursive_directory_iterator(
           dataset_dir, fs::directory_options::skip_permission_denied);
       it != fs::recursive_directory_iterator(); ++it) {
    if (const fs::directory_entry &entry = *it;
      entry.is_regular_file()) {
      if (const fs::path &entry_path = entry.path();
        entry_path.stem() == circuit_name
        && entry_path.extension() == ".qasm") {
        return std::make_tuple(
            fs::absolute(entry_path).parent_path(), circuit_name, ".qasm");
      }
    }
  }
  return std::nullopt;
}

std::optional<std::tuple<
  std::string, std::string,
  unsigned int, unsigned int, unsigned int> >
get_circuit_info(const std::string &circuit_file) {
  auto found_circuit = search_circuit(circuit_file);
  if (!found_circuit.has_value()) {
    return std::nullopt;
  }
  auto [circuit_folder, circuit_name, circuit_extension]
      = found_circuit.value();
  if (circuit_extension != ".qke") {
    return std::nullopt;
  }
  fs::path full_circuit_path
      = circuit_folder / (circuit_name + circuit_extension);
  std::string quake_module_text
      = readFileToString(full_circuit_path.string());
  auto [mlir_module, context_ptr] = extractMLIRContext(quake_module_text);
  return std::make_tuple(circuit_name, full_circuit_path,
                         getNumberOfQubits(FuncOp(mlir_module)),
                         getNumberOfGates(FuncOp(mlir_module)),
                         getCircuitDepth(FuncOp(mlir_module)));
}

void print_circuit_info(const std::string &circuit_file) {
  auto circuit_info = get_circuit_info(circuit_file);
  if (!circuit_info.has_value()) {
    auto found_circuit = search_circuit(circuit_file);
    if (!found_circuit.has_value()) {
      std::cout << "Circuit " + circuit_file + " not found." << std::endl;
      return;
    }
    auto [circuit_folder, circuit_name, circuit_extension]
        = found_circuit.value();
    if (circuit_extension == ".qasm") {
      const std::string qasm_to_quake_tool_path
          = std::string(MQSS_BUILD_DIR) + "/tools/qasm-to-quake";
      const std::string circuit_path
          = (circuit_folder / (circuit_name + circuit_extension)).string();
      std::cout << "Circuit " + circuit_name + " at " << circuit_path << ".\n"
          << "Run \"" << qasm_to_quake_tool_path << " --input " << circuit_path
          << " --output <output_file>.qke\" to convert it to Quake."
          << std::endl;
    }
    return;
  }
  auto [circuit_name, full_circuit_path, nr_qubits, nr_gates, depth]
      = circuit_info.value();
  std::cout << "Circuit " + circuit_name + ":" << std::endl;
  std::cout
      << "   Full circuit path: " << full_circuit_path << "\n"
      << "   Number of qubits: " << nr_qubits << "\n"
      << "   Number of gates: " << nr_gates << "\n"
      << "   Depth: " << depth << std::endl;
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
  if (fs::exists(quake_dataset_dir) && fs::is_directory(quake_dataset_dir)) {
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
    for (auto it = fs::recursive_directory_iterator(
             quake_dataset_dir, fs::directory_options::skip_permission_denied);
         it != fs::recursive_directory_iterator(); ++it) {
      if (const fs::directory_entry &entry = *it;
        entry.is_regular_file()) {
        if (const fs::path &entry_path = entry.path();
          entry_path.extension() == ".qke") {
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
      }
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
  return std::nullopt;
}


void print_dataset_info(const std::string &dataset_name) {
  if (dataset_name.empty()) {
    return;
  }
  fs::path quake_dataset_dir = fs::path(AI_DATASET_DIR)
                               / "Quake" / dataset_name;

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
  if (fs::exists(quake_dataset_dir) && fs::is_directory(quake_dataset_dir)) {
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