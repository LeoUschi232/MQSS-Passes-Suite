#ifndef STATISTICS_FOR_RQCG_HPP
#define STATISTICS_FOR_RQCG_HPP

// Support includes
#include "Support/mlir_utils.hpp"

// Standard library includes
#include <filesystem>

namespace fs = std::filesystem;

namespace ai_pass_selector {
/// Sizes
constexpr unsigned int GATES_WEIGHTS_SIZE = 37;
constexpr unsigned int OPERATIONS_SUBSET_SIZE = 34;
constexpr unsigned int MEASUREMENTS_SUBSET_SIZE = 3;
/// Indexes
constexpr unsigned int X_INDEX = 0;
constexpr unsigned int CX_INDEX = 1;
constexpr unsigned int CCX_INDEX = 2;
constexpr unsigned int C3plus_X_INDEX = 3;
constexpr unsigned int Y_INDEX = 4;
constexpr unsigned int controlled_Y_INDEX = 5;
constexpr unsigned int Z_INDEX = 6;
constexpr unsigned int controlled_Z_INDEX = 7;
constexpr unsigned int H_INDEX = 8;
constexpr unsigned int controlled_H_INDEX = 9;
constexpr unsigned int S_INDEX = 10;
constexpr unsigned int controlled_S_INDEX = 11;
constexpr unsigned int SDG_INDEX = 12;
constexpr unsigned int controlled_SDG_INDEX = 13;
constexpr unsigned int T_INDEX = 14;
constexpr unsigned int controlled_T_INDEX = 15;
constexpr unsigned int TDG_INDEX = 16;
constexpr unsigned int controlled_TDG_INDEX = 17;
constexpr unsigned int RX_INDEX = 18;
constexpr unsigned int controlled_RX_INDEX = 19;
constexpr unsigned int RY_INDEX = 20;
constexpr unsigned int controlled_RY_INDEX = 21;
constexpr unsigned int RZ_INDEX = 22;
constexpr unsigned int controlled_RZ_INDEX = 23;
constexpr unsigned int SWAP_INDEX = 24;
constexpr unsigned int controlled_SWAP_INDEX = 25;
constexpr unsigned int R1_INDEX = 26;
constexpr unsigned int controlled_R1_INDEX = 27;
constexpr unsigned int U2_INDEX = 28;
constexpr unsigned int controlled_U2_INDEX = 29;
constexpr unsigned int U3_INDEX = 30;
constexpr unsigned int controlled_U3_INDEX = 31;
constexpr unsigned int PHASED_RX_INDEX = 32;
constexpr unsigned int controlled_PHASED_RX_INDEX = 33;
constexpr unsigned int MX_INDEX = 34;
constexpr unsigned int MY_INDEX = 35;
constexpr unsigned int MZ_INDEX = 36;

/**
 *
 * @param dataset_name
 * @return
 */
std::optional<std::pair<std::tuple<double, double, double, double, double>,
                        std::array<unsigned int, GATES_WEIGHTS_SIZE>>>
extract_dataset_statistics(const std::string &dataset_name);

/**
 *
 * @param dataset_name
 */
void print_dataset_statistics(const std::string &dataset_name);

////////////////////////////////////////////////////////////////////////////////
/// Dataset statistics embedded as available C++ data.
///
constexpr std::tuple MQT_BENCH_QUBITS_AND_GATES_DISTRIBUTION_PARAMS = {
    /* mean_qubits */ 62.1292,
    /* mean_gates */ 5300.87,
    /* cholesky_L11 */ 38.598,
    /* cholesky_L21 */ 4260.97,
    /* cholesky_L22 */ 7922.95};

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

constexpr std::tuple
    RANDOMTEST_PREEMPTIVE_QUBITS_AND_GATES_DISTRIBUTION_PARAMS = {
        /* mean_qubits */ 4.95,
        /* mean_gates */ 21.1,
        /* cholesky_L11 */ 0.825578,
        /* cholesky_L21 */ 0.898892,
        /* cholesky_L22 */ 6.66668};

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
