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

std::optional<std::pair<std::array<double, CHOLESKY_PARAMS_SIZE>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
extract_dataset_statistics(const std::string &dataset_name) {
  std::vector<fs::path> files = get_dataset_files(dataset_name);
  if (files.empty()) {
    return std::nullopt;
  }
  unsigned int nr_files = files.size();
  std::vector<
      std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>>
      qubits_gates_operations_measurements;
  qubits_gates_operations_measurements.reserve(nr_files);
  std::array<unsigned int, GATES_WEIGHTS_SIZE> gates_weights{};

  bool all_measured = true;
  unsigned int progress = 0;
  for (auto entry_path : files) {
    updateProgress(++progress, nr_files,
                   "Extracting statistics from: " + dataset_name);
    std::string quake_module_text = readFileToString(entry_path.string());
    auto [circuit, context_ptr] = extractMLIRContext(quake_module_text);
    unsigned int nr_qubits = 0, nr_gates = 0, nr_operations = 0,
                 nr_measurements = 0;

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
      if (!isGate(op)) {
        return;
      }
      if (isMeasurement(op)) {
        if (isa<quake::MxOp>(op)) {
          gates_weights[to_index(GateWeightIndex::MX)]++;
        } else if (isa<quake::MyOp>(op)) {
          gates_weights[to_index(GateWeightIndex::MY)]++;
        } else if (isa<quake::MzOp>(op)) {
          gates_weights[to_index(GateWeightIndex::MZ)]++;
        } else {
          throw std::runtime_error("Measurement gate op not Mx|My|Mz.");
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
              nr_measurements++;
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
      nr_operations++;
      auto gate = dyn_cast<quake::OperatorInterface>(op);
      std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
      unsigned int nr_controls = controls.size();
      bool isAdj = gate.isAdj();
      if (isAdj && !isa<quake::SOp>(op) && !isa<quake::TOp>(op)) {
        std::cerr << "\nWarning: Spotted non-S or non-T adjoint gate possibly "
                     "inverting angle of rotation gates."
                  << std::endl;
      }

      switch (GATE_SYMBOL(getOnlyGateName(op))) {
      case GateSymbol::X:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::X)]++;
        } else if (nr_controls == 1) {
          gates_weights[to_index(GateWeightIndex::CX)]++;
        } else if (nr_controls == 2) {
          gates_weights[to_index(GateWeightIndex::CCX)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::C3PlusX)]++;
        }
        break;
      case GateSymbol::Y:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::Y)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledY)]++;
        }
        break;
      case GateSymbol::Z:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::Z)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledZ)]++;
        }
        break;
      case GateSymbol::H:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::H)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledH)]++;
        }
        break;
      case GateSymbol::S:
        if (nr_controls <= 0) {
          if (isAdj) {
            gates_weights[to_index(GateWeightIndex::SDG)]++;
          } else {
            gates_weights[to_index(GateWeightIndex::S)]++;
          }
        } else {
          if (isAdj) {
            gates_weights[to_index(GateWeightIndex::ControlledSDG)]++;
          } else {
            gates_weights[to_index(GateWeightIndex::ControlledS)]++;
          }
        }
        break;
      case GateSymbol::T:
        if (nr_controls <= 0) {
          if (isAdj) {
            gates_weights[to_index(GateWeightIndex::TDG)]++;
          } else {
            gates_weights[to_index(GateWeightIndex::T)]++;
          }
        } else {
          if (isAdj) {
            gates_weights[to_index(GateWeightIndex::ControlledTDG)]++;
          } else {
            gates_weights[to_index(GateWeightIndex::ControlledT)]++;
          }
        }
        break;
      case GateSymbol::RX:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::RX)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledRX)]++;
        }
        break;
      case GateSymbol::RY:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::RY)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledRY)]++;
        }
        break;
      case GateSymbol::RZ:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::RZ)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledRZ)]++;
        }
        break;
      case GateSymbol::SWAP:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::SWAP)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledSWAP)]++;
        }
        break;
      case GateSymbol::R1:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::R1)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledR1)]++;
        }
        break;
      case GateSymbol::U2:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::U2)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledU2)]++;
        }
        break;
      case GateSymbol::U3:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::U3)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledU3)]++;
        }
        break;
      case GateSymbol::PHASED_RX:
        if (nr_controls <= 0) {
          gates_weights[to_index(GateWeightIndex::PhasedRX)]++;
        } else {
          gates_weights[to_index(GateWeightIndex::ControlledPhasedRX)]++;
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
    if (nr_measurements != nr_qubits) {
      all_measured = false;
    }
    qubits_gates_operations_measurements.emplace_back(
        nr_qubits, nr_gates, nr_operations, nr_measurements);
  }
  std::cout << std::endl;
  unsigned int N = qubits_gates_operations_measurements.size();
  if (N <= 1) {
    return std::nullopt;
  }
  double mean_qubits = 0.0, mean_gates = 0.0, mean_operations = 0.0,
         mean_measurements = 0.0;
  for (const auto &[nr_qubits, nr_gates, nr_operations, nr_measurements] :
       qubits_gates_operations_measurements) {
    assert(nr_gates == nr_operations + nr_measurements &&
           "nr_gates!=nr_operations+nr_measurements");
    mean_qubits += nr_qubits;
    mean_gates += nr_gates;
    mean_operations += nr_operations;
    mean_measurements += nr_measurements;
  }
  mean_qubits /= N;
  mean_gates /= N;
  mean_operations /= N;
  mean_measurements /= N;
  if (mean_qubits < 2.0 || mean_operations < 2.0) {
    // Rasonable quanbtum circuits should have at least 2 qubits and at least 2
    // operations. If a dataset has most circuit with either only 1 qubit or 1
    // gate or is empty alltogether, that dataset is not worth analysing, using
    // or keeping.
    return std::nullopt;
  }
  double variance_qubits = 0.0;
  double variance_gates = 0.0;
  double covariance_gates = 0.0;
  double variance_operations = 0.0;
  double covariance_operations = 0.0;
  double variance_measurements = 0.0;
  double covariance_measurements = 0.0;
  for (const auto &[nr_qubits, nr_gates, nr_operations, nr_measurements] :
       qubits_gates_operations_measurements) {
    variance_qubits += (nr_qubits - mean_qubits) * (nr_qubits - mean_qubits);
    variance_gates += (nr_gates - mean_gates) * (nr_gates - mean_gates);
    covariance_gates += (nr_qubits - mean_qubits) * (nr_gates - mean_gates);
    variance_operations +=
        (nr_operations - mean_operations) * (nr_operations - mean_operations);
    covariance_operations +=
        (nr_qubits - mean_qubits) * (nr_operations - mean_operations);
    variance_measurements += (nr_measurements - mean_measurements) *
                             (nr_measurements - mean_measurements);
    covariance_measurements +=
        (nr_qubits - mean_qubits) * (nr_measurements - mean_measurements);
  }
  variance_qubits /= N - 1;
  variance_gates /= N - 1;
  covariance_gates /= N - 1;
  variance_operations /= N - 1;
  covariance_operations /= N - 1;
  variance_measurements /= N - 1;
  covariance_measurements /= N - 1;
  std::array<double, CHOLESKY_PARAMS_SIZE> cholesky_params{};
  cholesky_params[to_index(CholeskyParamIndex::MeanQubits)] = mean_qubits;
  cholesky_params[to_index(CholeskyParamIndex::MeanGates)] = mean_gates;
  cholesky_params[to_index(CholeskyParamIndex::MeanOperations)] =
      mean_operations;
  cholesky_params[to_index(CholeskyParamIndex::MeanMeasurements)] =
      mean_measurements;
  cholesky_params[to_index(CholeskyParamIndex::QubitsL11)] =
      std::sqrt(variance_qubits);

  if (variance_qubits <= 0.0) {
    cholesky_params[to_index(CholeskyParamIndex::GatesL21)] = 0.0;
    cholesky_params[to_index(CholeskyParamIndex::GatesL22)] =
        std::sqrt(variance_gates);
    cholesky_params[to_index(CholeskyParamIndex::OperationsL21)] = 0.0;
    cholesky_params[to_index(CholeskyParamIndex::OperationsL22)] =
        std::sqrt(variance_operations);
    cholesky_params[to_index(CholeskyParamIndex::MeasurementsL21)] = 0.0;
    cholesky_params[to_index(CholeskyParamIndex::MeasurementsL22)] =
        std::sqrt(variance_measurements);
    return std::make_pair(cholesky_params, gates_weights);
  }
  if (variance_gates <= 0.0) {
    cholesky_params[to_index(CholeskyParamIndex::GatesL21)] = 0.0;
    cholesky_params[to_index(CholeskyParamIndex::GatesL22)] = 0.0;
  } else if (double determinant = variance_qubits * variance_gates -
                                  covariance_gates * covariance_gates;
             determinant <= 0.0) {
    return std::nullopt;
  } else {
    cholesky_params[to_index(CholeskyParamIndex::GatesL21)] =
        covariance_gates /
        cholesky_params[to_index(CholeskyParamIndex::QubitsL11)];
    cholesky_params[to_index(CholeskyParamIndex::GatesL22)] =
        std::sqrt(variance_gates -
                  cholesky_params[to_index(CholeskyParamIndex::GatesL21)] *
                      cholesky_params[to_index(CholeskyParamIndex::GatesL21)]);
  }
  if (variance_operations <= 0.0) {
    cholesky_params[to_index(CholeskyParamIndex::OperationsL21)] = 0.0;
    cholesky_params[to_index(CholeskyParamIndex::OperationsL22)] = 0.0;
  } else if (double determinant = variance_qubits * variance_operations -
                                  covariance_operations * covariance_operations;
             determinant <= 0.0) {
    return std::nullopt;
  } else {
    cholesky_params[to_index(CholeskyParamIndex::OperationsL21)] =
        covariance_operations /
        cholesky_params[to_index(CholeskyParamIndex::QubitsL11)];
    cholesky_params[to_index(CholeskyParamIndex::OperationsL22)] = std::sqrt(
        variance_operations -
        cholesky_params[to_index(CholeskyParamIndex::OperationsL21)] *
            cholesky_params[to_index(CholeskyParamIndex::OperationsL21)]);
  }
  if (variance_measurements <= 0.0) {
    cholesky_params[to_index(CholeskyParamIndex::MeasurementsL21)] = 0.0;
    cholesky_params[to_index(CholeskyParamIndex::MeasurementsL22)] = 0.0;
  } else if (all_measured) {
    cholesky_params[to_index(CholeskyParamIndex::MeasurementsL21)] =
        cholesky_params[to_index(CholeskyParamIndex::QubitsL11)];
    cholesky_params[to_index(CholeskyParamIndex::MeasurementsL22)] = 0.0;
  } else if (double determinant =
                 variance_qubits * variance_measurements -
                 covariance_measurements * covariance_measurements;
             determinant <= 0.0) {
    return std::nullopt;
  } else {
    cholesky_params[to_index(CholeskyParamIndex::MeasurementsL21)] =
        covariance_measurements /
        cholesky_params[to_index(CholeskyParamIndex::QubitsL11)];
    cholesky_params[to_index(CholeskyParamIndex::MeasurementsL22)] = std::sqrt(
        variance_measurements -
        cholesky_params[to_index(CholeskyParamIndex::MeasurementsL21)] *
            cholesky_params[to_index(CholeskyParamIndex::MeasurementsL21)]);
  }
  return std::make_pair(cholesky_params, gates_weights);
}
void print_dataset_statistics(const std::string &dataset_name) {
  auto dataset_statistics = extract_dataset_statistics(dataset_name);
  if (!dataset_statistics.has_value()) {
    std::cerr << "Dataset \"" << dataset_name
              << "\" either does not exist or has weak statistics."
              << std::endl;
    return;
  }
  auto [cholesky_params, gates_weights] = dataset_statistics.value();
  std::cout << "dataset_name: \"" << dataset_name << "\"" << std::endl;
  std::cout << "cholesky_params: " << std::endl;
  std::cout << "  mean_qubits: "
            << cholesky_params[to_index(CholeskyParamIndex::MeanQubits)]
            << std::endl;
  std::cout << "  mean_gates: "
            << cholesky_params[to_index(CholeskyParamIndex::MeanGates)]
            << std::endl;
  std::cout << "  mean_operations: "
            << cholesky_params[to_index(CholeskyParamIndex::MeanOperations)]
            << std::endl;
  std::cout << "  mean_measurements: "
            << cholesky_params[to_index(CholeskyParamIndex::MeanMeasurements)]
            << std::endl;
  std::cout << "  qubits_L11: "
            << cholesky_params[to_index(CholeskyParamIndex::QubitsL11)]
            << std::endl;
  std::cout << "  gates_L21: "
            << cholesky_params[to_index(CholeskyParamIndex::GatesL21)]
            << std::endl;
  std::cout << "  gates_L22: "
            << cholesky_params[to_index(CholeskyParamIndex::GatesL22)]
            << std::endl;
  std::cout << "  operations_L21: "
            << cholesky_params[to_index(CholeskyParamIndex::OperationsL21)]
            << std::endl;
  std::cout << "  operations_L22: "
            << cholesky_params[to_index(CholeskyParamIndex::OperationsL22)]
            << std::endl;
  std::cout << "  measurements_L21: "
            << cholesky_params[to_index(CholeskyParamIndex::MeasurementsL21)]
            << std::endl;
  std::cout << "  measurements_L22: "
            << cholesky_params[to_index(CholeskyParamIndex::MeasurementsL22)]
            << std::endl;
  std::cout << "gates_weights:" << std::endl;
  std::cout << "  X: " << gates_weights[to_index(GateWeightIndex::X)]
            << std::endl;
  std::cout << "  CX: " << gates_weights[to_index(GateWeightIndex::CX)]
            << std::endl;
  std::cout << "  CCX: " << gates_weights[to_index(GateWeightIndex::CCX)]
            << std::endl;
  std::cout << "  C3plus_X: "
            << gates_weights[to_index(GateWeightIndex::C3PlusX)] << std::endl;
  std::cout << "  Y: " << gates_weights[to_index(GateWeightIndex::Y)]
            << std::endl;
  std::cout << "  controlled_Y: "
            << gates_weights[to_index(GateWeightIndex::ControlledY)]
            << std::endl;
  std::cout << "  Z: " << gates_weights[to_index(GateWeightIndex::Z)]
            << std::endl;
  std::cout << "  controlled_Z: "
            << gates_weights[to_index(GateWeightIndex::ControlledZ)]
            << std::endl;
  std::cout << "  H: " << gates_weights[to_index(GateWeightIndex::H)]
            << std::endl;
  std::cout << "  controlled_H: "
            << gates_weights[to_index(GateWeightIndex::ControlledH)]
            << std::endl;
  std::cout << "  S: " << gates_weights[to_index(GateWeightIndex::S)]
            << std::endl;
  std::cout << "  controlled_S: "
            << gates_weights[to_index(GateWeightIndex::ControlledS)]
            << std::endl;
  std::cout << "  SDG: " << gates_weights[to_index(GateWeightIndex::SDG)]
            << std::endl;
  std::cout << "  controlled_SDG: "
            << gates_weights[to_index(GateWeightIndex::ControlledSDG)]
            << std::endl;
  std::cout << "  T: " << gates_weights[to_index(GateWeightIndex::T)]
            << std::endl;
  std::cout << "  controlled_T: "
            << gates_weights[to_index(GateWeightIndex::ControlledT)]
            << std::endl;
  std::cout << "  TDG: " << gates_weights[to_index(GateWeightIndex::TDG)]
            << std::endl;
  std::cout << "  controlled_TDG: "
            << gates_weights[to_index(GateWeightIndex::ControlledTDG)]
            << std::endl;
  std::cout << "  RX: " << gates_weights[to_index(GateWeightIndex::RX)]
            << std::endl;
  std::cout << "  controlled_RX: "
            << gates_weights[to_index(GateWeightIndex::ControlledRX)]
            << std::endl;
  std::cout << "  RY: " << gates_weights[to_index(GateWeightIndex::RY)]
            << std::endl;
  std::cout << "  controlled_RY: "
            << gates_weights[to_index(GateWeightIndex::ControlledRY)]
            << std::endl;
  std::cout << "  RZ: " << gates_weights[to_index(GateWeightIndex::RZ)]
            << std::endl;
  std::cout << "  controlled_RZ: "
            << gates_weights[to_index(GateWeightIndex::ControlledRZ)]
            << std::endl;
  std::cout << "  SWAP: " << gates_weights[to_index(GateWeightIndex::SWAP)]
            << std::endl;
  std::cout << "  controlled_SWAP: "
            << gates_weights[to_index(GateWeightIndex::ControlledSWAP)]
            << std::endl;
  std::cout << "  R1: " << gates_weights[to_index(GateWeightIndex::R1)]
            << std::endl;
  std::cout << "  controlled_R1: "
            << gates_weights[to_index(GateWeightIndex::ControlledR1)]
            << std::endl;
  std::cout << "  U2: " << gates_weights[to_index(GateWeightIndex::U2)]
            << std::endl;
  std::cout << "  controlled_U2: "
            << gates_weights[to_index(GateWeightIndex::ControlledU2)]
            << std::endl;
  std::cout << "  U3: " << gates_weights[to_index(GateWeightIndex::U3)]
            << std::endl;
  std::cout << "  controlled_U3: "
            << gates_weights[to_index(GateWeightIndex::ControlledU3)]
            << std::endl;
  std::cout << "  PHASED_RX: "
            << gates_weights[to_index(GateWeightIndex::PhasedRX)] << std::endl;
  std::cout << "  controlled_PHASED_RX: "
            << gates_weights[to_index(GateWeightIndex::ControlledPhasedRX)]
            << std::endl;
  std::cout << "  MX: " << gates_weights[to_index(GateWeightIndex::MX)]
            << std::endl;
  std::cout << "  MY: " << gates_weights[to_index(GateWeightIndex::MY)]
            << std::endl;
  std::cout << "  MZ: " << gates_weights[to_index(GateWeightIndex::MZ)]
            << std::endl;
}

std::optional<std::pair<std::array<double, CHOLESKY_PARAMS_SIZE>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
get_precomputed_dataset_statistics(const std::string &dataset_name) {
  std::string dataset_name_lower = dataset_name;
  std::transform(dataset_name_lower.begin(), dataset_name_lower.end(),
                 dataset_name_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (dataset_name_lower == "mqtbench" || dataset_name_lower == "mqt-bench" ||
      dataset_name_lower == "mqt_bench") {
    return std::make_pair(MQT_BENCH_QUBITS_CHOLSEKY_PARAMS,
                          MQT_BENCH_GATES_WEIGHTS);
  }
  if (dataset_name_lower == "pyscf" || dataset_name_lower == "chemistry") {
    return std::make_pair(MQT_BENCH_QUBITS_CHOLSEKY_PARAMS,
                          CHEMISTRY_GATES_WEIGHTS);
  }
  std::cerr << "No embedded dataset statistics for dataset: " << dataset_name
            << std::endl;
  return std::nullopt;
}

} // namespace ai_pass_selector
