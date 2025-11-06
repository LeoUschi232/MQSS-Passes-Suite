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
 *
 * @param dataset_name
 * @return
 */
std::optional<std::pair<std::array<double, CHOLESKY_PARAMS_SIZE>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
extract_dataset_statistics(const std::string &dataset_name);

/**
 *
 * @param dataset_name
 */
void print_dataset_statistics(const std::string &dataset_name);

/**
 *
 * @param dataset_name
 * @return
 */
std::optional<std::pair<std::array<double, CHOLESKY_PARAMS_SIZE>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
get_precomputed_dataset_statistics(const std::string &dataset_name);

////////////////////////////////////////////////////////////////////////////////
/// Dataset statistics embedded as available C++ data.
constexpr std::array<double, CHOLESKY_PARAMS_SIZE>
    MQT_BENCH_QUBITS_CHOLSEKY_PARAMS = {
        /* mean_qubits */ 62.1292,
        /* mean_gates */ 5300.87,
        /* mean_operations */ 5238.94,
        /* mean_measurements */ 61.93,
        /* qubits_L11 */ 38.598,
        /* gates_L21 */ 4260.97,
        /* gates_L22 */ 7922.95,
        /* operations_L21 */ 4222.39,
        /* operations_L22 */ 7922.86,
        /* measurements_L21 */ 38.578,
        /* measurements_L22 */ 0.398984};

constexpr std::array<unsigned int, GATES_WEIGHTS_SIZE> MQT_BENCH_GATES_WEIGHTS =
    {/* X */ 17022u,
     /* CX */ 5623760u,
     /* CCX */ 34803u,
     /* C3plus_X */ 0u,
     /* Y */ 5712u,
     /* controlled_Y */ 16372u,
     /* Z */ 6018u,
     /* controlled_Z */ 44138u,
     /* H */ 344388u,
     /* controlled_H */ 16446u,
     /* S */ 43873u,
     /* controlled_S */ 0u,
     /* SDG */ 25324u,
     /* controlled_SDG */ 0u,
     /* T */ 37419u,
     /* controlled_T */ 0u,
     /* TDG */ 37417u,
     /* controlled_TDG */ 0u,
     /* RX */ 26867u,
     /* controlled_RX */ 16275u,
     /* RY */ 192001u,
     /* controlled_RY */ 17014u,
     /* RZ */ 73845u,
     /* controlled_RZ */ 17321u,
     /* SWAP */ 33686u,
     /* controlled_SWAP */ 16813u,
     /* R1 */ 878975u,
     /* controlled_R1 */ 2225341u,
     /* U2 */ 254545u,
     /* controlled_U2 */ 0u,
     /* U3 */ 141413u,
     /* controlled_U3 */ 32482u,
     /* PHASED_RX */ 0u,
     /* controlled_PHASED_RX */ 0u,
     /* MX */ 0u,
     /* MY */ 0u,
     /* MZ */ 120330u};

constexpr std::array<double, CHOLESKY_PARAMS_SIZE>
    CHEMISTRY_QUBITS_CHOLSEKY_PARAMS = {
        /* mean_qubits */ 13.907,
        /* mean_gates */ 387.052,
        /* mean_operations */ 373.145,
        /* mean_measurements */ 13.907,
        /* qubits_L11 */ 5.00308,
        /* gates_L21 */ 269.676,
        /* gates_L22 */ 127.196,
        /* operations_L21 */ 264.672,
        /* operations_L22 */ 127.196,
        /* measurements_L21 */ 5.00308,
        /* measurements_L22 */ 0.0};

constexpr std::array<unsigned int, GATES_WEIGHTS_SIZE> CHEMISTRY_GATES_WEIGHTS =
    {/* X */ 13792u,
     /* CX */ 297196u,
     /* CCX */ 0u,
     /* C3plus_X */ 0u,
     /* Y */ 0u,
     /* controlled_Y */ 0u,
     /* Z */ 0u,
     /* controlled_Z */ 0u,
     /* H */ 139496u,
     /* controlled_H */ 0u,
     /* S */ 0u,
     /* controlled_S */ 0u,
     /* SDG */ 0u,
     /* controlled_SDG */ 0u,
     /* T */ 0u,
     /* controlled_T */ 0u,
     /* TDG */ 0u,
     /* controlled_TDG */ 0u,
     /* RX */ 88744u,
     /* controlled_RX */ 0u,
     /* RY */ 776u,
     /* controlled_RY */ 0u,
     /* RZ */ 37624u,
     /* controlled_RZ */ 0u,
     /* SWAP */ 0u,
     /* controlled_SWAP */ 0u,
     /* R1 */ 0u,
     /* controlled_R1 */ 0u,
     /* U2 */ 0u,
     /* controlled_U2 */ 0u,
     /* U3 */ 0u,
     /* controlled_U3 */ 0u,
     /* PHASED_RX */ 0u,
     /* controlled_PHASED_RX */ 0u,
     /* MX */ 0u,
     /* MY */ 0u,
     /* MZ */ 21528u};

constexpr std::array<double, CHOLESKY_PARAMS_SIZE>
    RANDOMTEST_PREEMPTIVE_QUBITS_CHOLSEKY_PARAMS = {
        /* mean_qubits */ 4.95,
        /* mean_gates */ 21.1,
        /* mean_operations */ 16.15,
        /* mean_measurements */ 4.95,
        /* qubits_L11 */ 0.686333,
        /* gates_L21 */ 3.10575,
        /* gates_L22 */ 5.96363,
        /* operations_L21 */ 2.41942,
        /* operations_L22 */ 5.96363,
        /* measurements_L21 */ 0.686333,
        /* measurements_L22 */ 0.1};

constexpr std::array<unsigned int, GATES_WEIGHTS_SIZE>
    RANDOMTEST_PREEMPTIVE_GATES_WEIGHTS = {/* X */ 4u,
                                           /* CX */ 42u,
                                           /* CCX */ 2u,
                                           /* C3plus_X */ 3u,
                                           /* Y */ 1u,
                                           /* controlled_Y */ 7u,
                                           /* Z */ 1u,
                                           /* controlled_Z */ 3u,
                                           /* H */ 31u,
                                           /* controlled_H */ 0u,
                                           /* S */ 49u,
                                           /* controlled_S */ 1u,
                                           /* SDG */ 41u,
                                           /* controlled_SDG */ 5u,
                                           /* T */ 1u,
                                           /* controlled_T */ 0u,
                                           /* TDG */ 10u,
                                           /* controlled_TDG */ 1u,
                                           /* RX */ 36u,
                                           /* controlled_RX */ 4u,
                                           /* RY */ 25u,
                                           /* controlled_RY */ 10u,
                                           /* RZ */ 29u,
                                           /* controlled_RZ */ 3u,
                                           /* SWAP */ 3u,
                                           /* controlled_SWAP */ 2u,
                                           /* R1 */ 2u,
                                           /* controlled_R1 */ 0u,
                                           /* U2 */ 0u,
                                           /* controlled_U2 */ 1u,
                                           /* U3 */ 0u,
                                           /* controlled_U3 */ 0u,
                                           /* PHASED_RX */ 5u,
                                           /* controlled_PHASED_RX */ 1u,
                                           /* MX */ 8u,
                                           /* MY */ 13u,
                                           /* MZ */ 78u};
} // namespace ai_pass_selector

#endif // STATISTICS_FOR_RQCG_HPP
