#ifndef STATISTICS_FOR_RQCG_HPP
#define STATISTICS_FOR_RQCG_HPP

// Support includes
#include "Support/mlir_utils.hpp"

// Standard library includes
#include <cstddef>
#include <filesystem>

namespace fs = std::filesystem;

namespace ai_pass_selector {
/// Sizes
constexpr unsigned int GATES_WEIGHTS_SIZE = 37;
constexpr unsigned int OPERATIONS_SUBSET_SIZE = 34;
constexpr unsigned int CHOLESKY_PARAMS_SIZE = 11;
/// Gates Indexes
enum class GateWeightIndex : unsigned int {
  X = 0,
  CX = 1,
  CCX = 2,
  C3PlusX = 3,
  Y = 4,
  ControlledY = 5,
  Z = 6,
  ControlledZ = 7,
  H = 8,
  ControlledH = 9,
  S = 10,
  ControlledS = 11,
  SDG = 12,
  ControlledSDG = 13,
  T = 14,
  ControlledT = 15,
  TDG = 16,
  ControlledTDG = 17,
  RX = 18,
  ControlledRX = 19,
  RY = 20,
  ControlledRY = 21,
  RZ = 22,
  ControlledRZ = 23,
  SWAP = 24,
  ControlledSWAP = 25,
  R1 = 26,
  ControlledR1 = 27,
  U2 = 28,
  ControlledU2 = 29,
  U3 = 30,
  ControlledU3 = 31,
  PhasedRX = 32,
  ControlledPhasedRX = 33,
  MX = 34,
  MY = 35,
  MZ = 36
};

/// Cholseky Indexes
enum class CholeskyParamIndex : unsigned int {
  MeanQubits = 0,
  MeanGates = 1,
  MeanOperations = 2,
  MeanMeasurements = 3,
  QubitsL11 = 4,
  GatesL21 = 5,
  GatesL22 = 6,
  OperationsL21 = 7,
  OperationsL22 = 8,
  MeasurementsL21 = 9,
  MeasurementsL22 = 10
};

constexpr std::size_t to_index(GateWeightIndex index) {
  return static_cast<std::size_t>(index);
}

constexpr std::size_t to_index(CholeskyParamIndex index) {
  return static_cast<std::size_t>(index);
}

/**
 * @param dataset_name
 * @return
 */
std::optional<std::pair<std::array<double, CHOLESKY_PARAMS_SIZE>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
extract_dataset_statistics(const std::string &dataset_name);

/**
 * @param dataset_name
 */
void print_dataset_statistics(const std::string &dataset_name);

/**
 * @param statistics_yaml_file
 * @return [cholesky_params, gates_weights]
 */
std::optional<std::pair<std::array<double, CHOLESKY_PARAMS_SIZE>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
get_dataset_statistics_from_yaml_file(const fs::path &statistics_yaml_file);

/**
 * @param dataset_name
 * @return [cholesky_params, gates_weights]
 */
std::optional<std::pair<std::array<double, CHOLESKY_PARAMS_SIZE>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
get_dataset_statistics_from_dataset_name(const std::string &dataset_name);
} // namespace ai_pass_selector

#endif // STATISTICS_FOR_RQCG_HPP
