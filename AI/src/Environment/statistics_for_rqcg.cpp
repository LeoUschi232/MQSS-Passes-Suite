#include "Environment/statistics_for_rqcg.hpp"

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
#include "Utils/info_utils.hpp"
#include "Utils/progress_bar.hpp"
#include "Utils/tensor_utils.hpp"

// Standard library includes
#include <filesystem>
#include <iostream>
#include <random>
#include <unordered_set>

namespace fs = std::filesystem;
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {

std::optional<std::pair<std::tuple<double, double, double, double, double>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
extract_dataset_statistics(const std::string &dataset_name) {
  std::vector<fs::path> files = get_dataset_files(dataset_name);
  if (files.empty()) {
    return std::nullopt;
  }
  unsigned int nr_files = files.size();
  std::vector<std::pair<unsigned int, unsigned int>> qubits_and_gates;
  qubits_and_gates.reserve(nr_files);
  std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights{};

  unsigned int progress = 0;
  for (auto entry_path : files) {
    updateProgress(++progress, nr_files,
                   "Extracting statistics from: " + dataset_name);
    std::string quake_module_text = readFileToString(entry_path.string());
    auto [circuit, context_ptr] = extractMLIRContext(quake_module_text);
    unsigned int nr_qubits = 0;
    unsigned int nr_gates = 0;
    circuit.walk([&](Operation *op) {
      if (isa<quake::AllocaOp>(op)) {
        if (auto allocOp = dyn_cast<quake::AllocaOp>(op);
            allocOp.getType().dyn_cast<quake::RefType>()) {
          nr_qubits += 1;
        } else if (auto qvecType =
                       allocOp.getType().dyn_cast<quake::VeqType>()) {
          nr_qubits += qvecType.getSize();
        }
        return;
      }
      if (!isOperatingGate(op)) {
        return;
      }
      if (isMeasurementGate(op)) {
        if (isa<quake::MxOp>(op)) {
          gates_weights[MX_INDEX]++;
        } else if (isa<quake::MyOp>(op)) {
          gates_weights[MY_INDEX]++;
        } else if (isa<quake::MzOp>(op)) {
          gates_weights[MZ_INDEX]++;
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
      std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
      unsigned int nr_controls = controls.size();
      bool isAdj = gate.isAdj();
      if (isAdj && !isa<quake::SOp>(op) && !isa<quake::TOp>(op)) {
        std::cerr << "\nWarning: Spotted non-S or non-T adjoint gate possibly "
                     "inverting angle of rotation gates."
                  << std::endl;
      }

      switch (GATE_INDEX(getOnlyGateName(op))) {
      case X:
        if (nr_controls <= 0) {
          gates_weights[X_INDEX]++;
        } else if (nr_controls == 1) {
          gates_weights[CX_INDEX]++;
        } else if (nr_controls == 2) {
          gates_weights[CCX_INDEX]++;
        } else {
          gates_weights[C3plus_X_INDEX]++;
        }
        break;
      case Y:
        if (nr_controls <= 0) {
          gates_weights[Y_INDEX]++;
        } else {
          gates_weights[controlled_Y_INDEX]++;
        }
        break;
      case Z:
        if (nr_controls <= 0) {
          gates_weights[Z_INDEX]++;
        } else {
          gates_weights[controlled_Z_INDEX]++;
        }
        break;
      case H:
        if (nr_controls <= 0) {
          gates_weights[H_INDEX]++;
        } else {
          gates_weights[controlled_H_INDEX]++;
        }
        break;
      case S:
        if (nr_controls <= 0) {
          if (isAdj) {
            gates_weights[SDG_INDEX]++;
          } else {
            gates_weights[S_INDEX]++;
          }
        } else {
          if (isAdj) {
            gates_weights[controlled_SDG_INDEX]++;
          } else {
            gates_weights[controlled_S_INDEX]++;
          }
        }
        break;
      case T:
        if (nr_controls <= 0) {
          if (isAdj) {
            gates_weights[TDG_INDEX]++;
          } else {
            gates_weights[T_INDEX]++;
          }
        } else {
          if (isAdj) {
            gates_weights[controlled_TDG_INDEX]++;
          } else {
            gates_weights[controlled_T_INDEX]++;
          }
        }
        break;
      case RX:
        if (nr_controls <= 0) {
          gates_weights[RX_INDEX]++;
        } else {
          gates_weights[controlled_RX_INDEX]++;
        }
        break;
      case RY:
        if (nr_controls <= 0) {
          gates_weights[RY_INDEX]++;
        } else {
          gates_weights[controlled_RY_INDEX]++;
        }
        break;
      case RZ:
        if (nr_controls <= 0) {
          gates_weights[RZ_INDEX]++;
        } else {
          gates_weights[controlled_RZ_INDEX]++;
        }
        break;
      case SWAP:
        if (nr_controls <= 0) {
          gates_weights[SWAP_INDEX]++;
        } else {
          gates_weights[controlled_SWAP_INDEX]++;
        }
        break;
      case R1:
        if (nr_controls <= 0) {
          gates_weights[R1_INDEX]++;
        } else {
          gates_weights[controlled_R1_INDEX]++;
        }
        break;
      case U2:
        if (nr_controls <= 0) {
          gates_weights[U2_INDEX]++;
        } else {
          gates_weights[controlled_U2_INDEX]++;
        }
        break;
      case U3:
        if (nr_controls <= 0) {
          gates_weights[U3_INDEX]++;
        } else {
          gates_weights[controlled_U3_INDEX]++;
        }
        break;
      case PHASED_RX:
        if (nr_controls <= 0) {
          gates_weights[PHASED_RX_INDEX]++;
        } else {
          gates_weights[controlled_PHASED_RX_INDEX]++;
        }
        break;
      default:
        // Measurement gates are handled above.
        throw std::runtime_error("Unsupported gate in dataset.");
      }
    });
    if (nr_qubits < 2 || nr_gates < 2) {
      std::cerr << "\nWarning: Circuit with perceived 0 or 1 qubits or gates: "
                << entry_path << std::endl;
    }
    qubits_and_gates.emplace_back(nr_qubits, nr_gates);
  }
  std::cout << std::endl;
  unsigned int N = qubits_and_gates.size();
  if (N <= 1) {
    return std::nullopt;
  }
  double mean_qubits = 0.0;
  double mean_gates = 0.0;
  for (const auto &[qubits, gates] : qubits_and_gates) {
    mean_qubits += qubits;
    mean_gates += gates;
  }
  mean_qubits /= N;
  mean_gates /= N;
  double var_qubits = 0.0;
  double var_gates = 0.0;
  double covariance = 0.0;
  for (const auto &[qubits, gates] : qubits_and_gates) {
    var_qubits += (qubits - mean_qubits) * (qubits - mean_qubits);
    var_gates += (gates - mean_gates) * (gates - mean_gates);
    covariance += (qubits - mean_qubits) * (gates - mean_gates);
  }
  var_qubits /= N - 1;
  var_gates /= N - 1;
  covariance /= N - 1;
  if (var_qubits <= 0.0) {
    return std::make_pair(std::make_tuple(mean_qubits, mean_gates, 0.0, 0.0,
                                          std::sqrt(var_gates)),
                          gates_weights);
  }
  if (var_gates <= 0.0) {
    return std::make_pair(std::make_tuple(mean_qubits, mean_gates,
                                          std::sqrt(var_qubits), 0.0, 0.0),
                          gates_weights);
  }
  if (mean_qubits < 2.0 || mean_gates < 2.0) {
    // Rasonable quanbtum circuits should have at least 2 qubits and at least 2
    // gates. If a dataset has most circuit with either only 1 qubit or 1 gate
    // or is empty alltogether, that dataset is not worth analysing, using or
    // keeping.
    return std::nullopt;
  }
  if (double determinant = var_qubits * var_gates - covariance * covariance;
      determinant <= 0.0) {
    return std::nullopt;
  }
  double L11 = std::sqrt(var_qubits);
  double L21 = covariance / L11;
  double L22 = std::sqrt(var_gates - L21 * L21);
  return std::make_pair(std::make_tuple(mean_qubits, mean_gates, L11, L21, L22),
                        gates_weights);
}
void print_dataset_statistics(const std::string &dataset_name) {
  auto dataset_statistics = extract_dataset_statistics(dataset_name);
  if (!dataset_statistics.has_value()) {
    std::cerr << "Dataset \"" << dataset_name
              << "\" either does not exist or has weak statistics."
              << std::endl;
    return;
  }
  auto [qubits_and_gates_distribution_params, gates_weights] =
      dataset_statistics.value();
  std::cout << "dataset_name: \"" << dataset_name << "\"" << std::endl;
  std::cout << "qubits_and_gates_distribution_params: " << std::endl;
  auto [mean_qubits, mean_gates, L11, L21, L22] =
      qubits_and_gates_distribution_params;
  std::cout << "  mean_qubits: " << mean_qubits << std::endl;
  std::cout << "  mean_gates: " << mean_gates << std::endl;
  std::cout << "  cholesky_L11: " << L11 << std::endl;
  std::cout << "  cholesky_L21: " << L21 << std::endl;
  std::cout << "  cholesky_L22: " << L22 << std::endl;
  std::cout << "gates_weights:" << std::endl;
  std::cout << "  X: " << gates_weights[X_INDEX] << std::endl;
  std::cout << "  CX: " << gates_weights[CX_INDEX] << std::endl;
  std::cout << "  CCX: " << gates_weights[CCX_INDEX] << std::endl;
  std::cout << "  C3plus_X: " << gates_weights[C3plus_X_INDEX] << std::endl;
  std::cout << "  Y: " << gates_weights[Y_INDEX] << std::endl;
  std::cout << "  controlled_Y: " << gates_weights[controlled_Y_INDEX]
            << std::endl;
  std::cout << "  Z: " << gates_weights[Z_INDEX] << std::endl;
  std::cout << "  controlled_Z: " << gates_weights[controlled_Z_INDEX]
            << std::endl;
  std::cout << "  H: " << gates_weights[H_INDEX] << std::endl;
  std::cout << "  controlled_H: " << gates_weights[controlled_H_INDEX]
            << std::endl;
  std::cout << "  S: " << gates_weights[S_INDEX] << std::endl;
  std::cout << "  controlled_S: " << gates_weights[controlled_S_INDEX]
            << std::endl;
  std::cout << "  SDG: " << gates_weights[SDG_INDEX] << std::endl;
  std::cout << "  controlled_SDG: " << gates_weights[controlled_SDG_INDEX]
            << std::endl;
  std::cout << "  T: " << gates_weights[T_INDEX] << std::endl;
  std::cout << "  controlled_T: " << gates_weights[controlled_T_INDEX]
            << std::endl;
  std::cout << "  TDG: " << gates_weights[TDG_INDEX] << std::endl;
  std::cout << "  controlled_TDG: " << gates_weights[controlled_TDG_INDEX]
            << std::endl;
  std::cout << "  RX: " << gates_weights[RX_INDEX] << std::endl;
  std::cout << "  controlled_RX: " << gates_weights[controlled_RX_INDEX]
            << std::endl;
  std::cout << "  RY: " << gates_weights[RY_INDEX] << std::endl;
  std::cout << "  controlled_RY: " << gates_weights[controlled_RY_INDEX]
            << std::endl;
  std::cout << "  RZ: " << gates_weights[RZ_INDEX] << std::endl;
  std::cout << "  controlled_RZ: " << gates_weights[controlled_RZ_INDEX]
            << std::endl;
  std::cout << "  SWAP: " << gates_weights[SWAP_INDEX] << std::endl;
  std::cout << "  controlled_SWAP: " << gates_weights[controlled_SWAP_INDEX]
            << std::endl;
  std::cout << "  R1: " << gates_weights[R1_INDEX] << std::endl;
  std::cout << "  controlled_R1: " << gates_weights[controlled_R1_INDEX]
            << std::endl;
  std::cout << "  U2: " << gates_weights[U2_INDEX] << std::endl;
  std::cout << "  controlled_U2: " << gates_weights[controlled_U2_INDEX]
            << std::endl;
  std::cout << "  U3: " << gates_weights[U3_INDEX] << std::endl;
  std::cout << "  controlled_U3: " << gates_weights[controlled_U3_INDEX]
            << std::endl;
  std::cout << "  PHASED_RX: " << gates_weights[PHASED_RX_INDEX] << std::endl;
  std::cout << "  controlled_PHASED_RX: "
            << gates_weights[controlled_PHASED_RX_INDEX] << std::endl;
  std::cout << "  MX: " << gates_weights[MX_INDEX] << std::endl;
  std::cout << "  MY: " << gates_weights[MY_INDEX] << std::endl;
  std::cout << "  MZ: " << gates_weights[MZ_INDEX] << std::endl;
}

std::optional<std::pair<std::tuple<double, double, double, double, double>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
get_embedded_dataset_statistics(const std::string &dataset_name) {
  std::string dataset_name_lower = dataset_name;
  std::transform(dataset_name_lower.begin(), dataset_name_lower.end(),
                 dataset_name_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (dataset_name_lower == "mqtbench" || dataset_name_lower == "mqt-bench" ||
      dataset_name_lower == "mqt_bench") {
    return std::make_pair(MQT_BENCH_QUBITS_AND_GATES_DISTRIBUTION_PARAMS,
                          MQT_BENCH_GATES_WEIGHTS);
  }
  if (dataset_name_lower == "pyscf" || dataset_name_lower == "chemistry") {
    return std::make_pair(CHEMISTRY_QUBITS_AND_GATES_DISTRIBUTION_PARAMS,
                          CHEMISTRY_GATES_WEIGHTS);
  }
  std::cerr << "No embedded dataset statistics for dataset: " << dataset_name
            << std::endl;
  return std::nullopt;
}

} // namespace ai_pass_selector
