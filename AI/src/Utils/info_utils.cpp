#include "Utils/info_utils.hpp"

// Support includes
#include "Support/mlir_utils.hpp"

#include <Utils/progress_bar.hpp>

////////////////////////////////////////////////////////////////////////////////
/// The usages of llvm functions must come before the QuakeOps header which
/// expects them.
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
////////////////////////////////////////////////////////////////////////////////

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// Cudaq includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

// Utils includes
#include "Utils/tensor_utils.hpp"

// Standard library includes
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>

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
  std::string dataset_name_lower = dataset_name;
  std::transform(dataset_name_lower.begin(), dataset_name_lower.end(),
                 dataset_name_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::vector<fs::path> files;
  fs::path quake_dataset_dir;
  if (dataset_name_lower == "all" || dataset_name == "*") {
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
  unsigned int min_nr_qubits = max_int, min_nr_gates = max_int,
               min_depth = max_int;
  unsigned int max_nr_qubits = 0, max_nr_gates = 0, max_depth = 0,
               total_nr_qubits = 0, total_nr_gates = 0, total_depth = 0,
               nr_mx_gates = 0, nr_my_gates = 0, nr_mz_gates = 0;
  std::unordered_set<std::string> gates_with_more_than_1_targets = {};
  std::unordered_set<std::string> gates_with_more_than_1_controls = {};
  unsigned int nr_gates_with_more_than_1_targets = 0,
               nr_gates_with_more_than_1_controls = 0, most_targets = 0,
               most_controls = 0;
  std::array<unsigned int, 4> nr_x_gates{0, 0, 0, 0};
  std::array<unsigned int, 2> nr_y_gates{0, 0}, nr_z_gates{0, 0},
      nr_h_gates{0, 0}, nr_rx_gates{0, 0}, nr_ry_gates{0, 0}, nr_rz_gates{0, 0},
      nr_swap_gates{0, 0}, nr_r1_gates{0, 0}, nr_u2_gates{0, 0},
      nr_u3_gates{0, 0}, nr_phased_rx_gates{0, 0};
  std::array<unsigned int, 4> nr_s_gates{0, 0, 0, 0}, nr_t_gates{0, 0, 0, 0};

  unsigned int progress = 0;
  for (auto entry_path : files) {
    updateProgress(++progress, nr_files, "Retrieving info " + dataset_name);
    std::string quake_module_text = readFileToString(entry_path.string());
    auto [circuit, context_ptr] = extractMLIRContext(quake_module_text);
    unsigned int nr_qubits = 0;
    unsigned int nr_gates = 0;
    std::vector<unsigned int> depths;
    circuit.walk([&](Operation *op) {
      if (isa<quake::AllocaOp>(op)) {
        if (auto allocOp = dyn_cast<quake::AllocaOp>(op);
            allocOp.getType().dyn_cast<quake::RefType>()) {
          nr_qubits += 1;
        } else if (auto qvecType =
                       allocOp.getType().dyn_cast<quake::VeqType>()) {
          nr_qubits += qvecType.getSize();
        }
        depths.resize(nr_qubits, 0);
        return;
      }
      if (!isOperatingGate(op)) {
        return;
      }
      if (isMeasurementGate(op)) {
        if (isa<quake::MxOp>(op)) {
          nr_mx_gates++;
        } else if (isa<quake::MyOp>(op)) {
          nr_my_gates++;
        } else if (isa<quake::MzOp>(op)) {
          nr_mz_gates++;
        }
        for (auto operand : op->getOperands()) {
          if (operand.getType().isa<quake::RefType>()) {
            auto qubitIndexOpt =
                extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
            if (!qubitIndexOpt.has_value()) {
              continue;
            }
            if (int qubitIndex = qubitIndexOpt.value();
                0 <= qubitIndex && qubitIndex < nr_qubits) {
              nr_gates++;
              depths[qubitIndex]++;
            }
          } else {
            // Allocations are allowed to be vectorized but applications must be
            // references.
            throw std::runtime_error(
                "Measurement gate op has unsupported operand.");
          }
        }
        return;
      }
      nr_gates++;
      auto gate = dyn_cast<quake::OperatorInterface>(op);
      std::string gate_name = getOnlyGateName(op);
      int gate_index = GATE_INDEX(gate_name);
      std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
      most_targets =
          std::max(most_targets, static_cast<unsigned int>(targets.size()));
      if (targets.size() > 1) {
        gates_with_more_than_1_targets.insert(gate_name);
        nr_gates_with_more_than_1_targets++;
      }
      std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
      unsigned int count_index = controls.empty() ? 0 : 1;
      if (gate_index == X) {
        count_index = std::min(static_cast<unsigned int>(controls.size()), 3u);
      }
      most_controls =
          std::max(most_controls, static_cast<unsigned int>(controls.size()));
      if (controls.size() > 1) {
        gates_with_more_than_1_controls.insert(gate_name);
        nr_gates_with_more_than_1_controls++;
      }
      targets.insert(targets.end(), controls.begin(), controls.end());
      unsigned int local_max_depth = 0;
      for (int qubit : targets) {
        local_max_depth = std::max(local_max_depth, depths[qubit]);
      }
      for (int qubit : targets) {
        depths[qubit] = local_max_depth + 1;
      }
      if (gate.isAdj()) {
        if (isa<quake::SOp>(op) || isa<quake::TOp>(op)) {
          count_index += 2;
        } else {
          throw std::runtime_error("Only S and T gates can be daggered.");
        }
      }
      switch (gate_index) {
      case X:
        nr_x_gates[count_index]++;
        break;
      case Y:
        nr_y_gates[count_index]++;
        break;
      case Z:
        nr_z_gates[count_index]++;
        break;
      case H:
        nr_h_gates[count_index]++;
        break;
      case S:
        nr_s_gates[count_index]++;
        break;
      case T:
        nr_t_gates[count_index]++;
        break;
      case RX:
        nr_rx_gates[count_index]++;
        break;
      case RY:
        nr_ry_gates[count_index]++;
        break;
      case RZ:
        nr_rz_gates[count_index]++;
        break;
      case SWAP:
        nr_swap_gates[count_index]++;
        break;
      case R1:
        nr_r1_gates[count_index]++;
        break;
      case U2:
        nr_u2_gates[count_index]++;
        break;
      case U3:
        nr_u3_gates[count_index]++;
        break;
      case PHASED_RX:
        nr_phased_rx_gates[count_index]++;
        break;
      default:
        throw std::runtime_error("Unsupported gate in dataset.");
      }
    });
    nr_circuits++;
    min_nr_qubits = std::min(min_nr_qubits, nr_qubits);
    max_nr_qubits = std::max(max_nr_qubits, nr_qubits);
    total_nr_qubits += nr_qubits;
    min_nr_gates = std::min(min_nr_gates, nr_gates);
    max_nr_gates = std::max(max_nr_gates, nr_gates);
    total_nr_gates += nr_gates;
    unsigned int depth =
        depths.empty() ? 0u : *std::max_element(depths.begin(), depths.end());
    min_depth = std::min(min_depth, depth);
    max_depth = std::max(max_depth, depth);
    total_depth += depth;
    if (nr_qubits < 2) {
      std::cout << "\nCircuit with perceived 0 or 1 qubits: " << entry_path
                << std::endl;
    }
    if (nr_gates < 2) {
      std::cout << "\nCircuit with perceived 0 or 1 gates: " << entry_path
                << std::endl;
    }
    if (depth < 2) {
      std::cout << "\nCircuit with perceived 0 or 1 depth: " << entry_path
                << std::endl;
    }
  }
  std::cout << std::endl;
  std::vector<std::pair<std::string, std::string>> info;
  info.emplace_back("Dataset name", dataset_name);
  info.emplace_back("Number of circuits", std::to_string(nr_circuits));
  info.emplace_back("Minimum number of qubits", std::to_string(min_nr_qubits));
  info.emplace_back("Maximum number of qubits", std::to_string(max_nr_qubits));
  info.emplace_back("Total number of qubits", std::to_string(total_nr_qubits));
  info.emplace_back("Minimum number of gates", std::to_string(min_nr_gates));
  info.emplace_back("Maximum number of gates", std::to_string(max_nr_gates));
  info.emplace_back("Total number of gates", std::to_string(total_nr_gates));
  info.emplace_back("Minimum depth", std::to_string(min_depth));
  info.emplace_back("Maximum depth", std::to_string(max_depth));
  info.emplace_back("Total depth", std::to_string(total_depth));
  info.emplace_back("Number of Mx gates", std::to_string(nr_mx_gates));
  info.emplace_back("Number of My gates", std::to_string(nr_my_gates));
  info.emplace_back("Number of Mz gates", std::to_string(nr_mz_gates));
  info.emplace_back("Number of X gates", std::to_string(nr_x_gates[0]));
  info.emplace_back("Number of CX gates", std::to_string(nr_x_gates[1]));
  info.emplace_back("Number of CCX gates", std::to_string(nr_x_gates[2]));
  info.emplace_back("Number of (3+)-controlled-X gates",
                    std::to_string(nr_x_gates[3]));
  info.emplace_back("Number of Y gates", std::to_string(nr_y_gates[0]));
  info.emplace_back("Number of controlled-Y gates",
                    std::to_string(nr_y_gates[1]));
  info.emplace_back("Number of Z gates", std::to_string(nr_z_gates[0]));
  info.emplace_back("Number of controlled-Z gates",
                    std::to_string(nr_z_gates[1]));
  info.emplace_back("Number of H gates", std::to_string(nr_h_gates[0]));
  info.emplace_back("Number of controlled-H gates",
                    std::to_string(nr_h_gates[1]));
  info.emplace_back("Number of S gates", std::to_string(nr_s_gates[0]));
  info.emplace_back("Number of controlled-S gates",
                    std::to_string(nr_s_gates[1]));
  info.emplace_back("Number of Sdg gates", std::to_string(nr_s_gates[2]));
  info.emplace_back("Number of controlled-Sdg gates",
                    std::to_string(nr_s_gates[3]));
  info.emplace_back("Number of T gates", std::to_string(nr_t_gates[0]));
  info.emplace_back("Number of controlled-T gates",
                    std::to_string(nr_t_gates[1]));
  info.emplace_back("Number of Tdg gates", std::to_string(nr_t_gates[2]));
  info.emplace_back("Number of controlled-Tdg gates",
                    std::to_string(nr_t_gates[3]));
  info.emplace_back("Number of Rx gates", std::to_string(nr_rx_gates[0]));
  info.emplace_back("Number of controlled-Rx gates",
                    std::to_string(nr_rx_gates[1]));
  info.emplace_back("Number of Ry gates", std::to_string(nr_ry_gates[0]));
  info.emplace_back("Number of controlled-Ry gates",
                    std::to_string(nr_ry_gates[1]));
  info.emplace_back("Number of Rz gates", std::to_string(nr_rz_gates[0]));
  info.emplace_back("Number of controlled-Rz gates",
                    std::to_string(nr_rz_gates[1]));
  info.emplace_back("Number of Swap gates", std::to_string(nr_swap_gates[0]));
  info.emplace_back("Number of controlled-Swap gates",
                    std::to_string(nr_swap_gates[1]));
  info.emplace_back("Number of R1 gates", std::to_string(nr_r1_gates[0]));
  info.emplace_back("Number of controlled-R1 gates",
                    std::to_string(nr_r1_gates[1]));
  info.emplace_back("Number of U2 gates", std::to_string(nr_u2_gates[0]));
  info.emplace_back("Number of controlled-U2 gates",
                    std::to_string(nr_u2_gates[1]));
  info.emplace_back("Number of U3 gates", std::to_string(nr_u3_gates[0]));
  info.emplace_back("Number of controlled-U3 gates",
                    std::to_string(nr_u3_gates[1]));
  info.emplace_back("Number of PhasedRx gates",
                    std::to_string(nr_phased_rx_gates[0]));
  info.emplace_back("Number of controlled-PhasedRx gates",
                    std::to_string(nr_phased_rx_gates[1]));
  info.emplace_back("Number of gates with more than 1 target",
                    std::to_string(nr_gates_with_more_than_1_targets));
  info.emplace_back("Gates with more than 1 target",
                    setToString(gates_with_more_than_1_targets));
  info.emplace_back("Number of gates with more than 1 control",
                    std::to_string(nr_gates_with_more_than_1_controls));
  info.emplace_back("Gates with more than 1 control",
                    setToString(gates_with_more_than_1_controls));
  info.emplace_back("Most targets in a gate", std::to_string(most_targets));
  info.emplace_back("Most controls in a gate", std::to_string(most_controls));
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
  (void)agent_name;
  std::cerr << "Function print_agent_info not implemented yet" << std::endl;
}

} // namespace ai_pass_selector