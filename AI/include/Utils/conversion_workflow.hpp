#ifndef CONVERSION_WORKFLOW_HPP
#define CONVERSION_WORKFLOW_HPP

// MLIR includes
#include "mlir/Pass/Pass.h"

// Stdandard library includes
#include <filesystem>
#include <string>

namespace ai_pass_selector {

/**
 * Convert all Qasm files in the AI/Dataset/Qasm directory to Quake files
 * int the AI/Dataset/Quake directory.
 */
void convertAllQasmDatasetsToQuake();

/**
 * Convert all Qasm files in a given subdirectory if AI/Dataset/Qasm to Quake
 * files inside AI/Dataset/Quake.
 * @param subdirectory The subdirectory inside AI/Dataset/Qasm to convert
 * the Qasm files form.
 * @return 0 on success, -1 on failure.
 */
int convertQasmDatasetToQuake(const std::string &subdirectory);

/**
 * Convert all pass test circuits in the AI/Dataset/Quake/Passtest directory
 * to LaTeX files in the AI/Dataset/Latex/Passtest directory.
 * This function iterates over all pass test circuits, applies the
 * decompositions and transformations defined in the MQSS passes,
 * and generates LaTeX files for each circuit.
 */
void convertAllPasstestCircuitsToTikz();

/**
 * Convert q pass test circuit in the AI/Dataset/Quake/Passtest directory
 * to LaTeX files in the AI/Dataset/Latex/Passtest directory.
 * @param passname Name of the pass to be applied to the circuits.
 * @param pass The MLIR pass to be applied to the circuits.
 * @return 0 on success, -1 on failure.
 */
int convertPasstestCircuitToTikz(std::string passname,
                                 std::unique_ptr<mlir::Pass> pass);

constexpr int TENSORTEST_MAX_QUBITS = 5;
constexpr int TENSORTEST_MAX_INSTRUCTIONS = 10;
constexpr int TENSORTEST_MAX_DEPTH = 10;

/**
 *
 */
void convertAllTensortestCircuitsToTikz();

/**
 *
 * @param circuit_name
 * @return
 */
int convertTensortestCircuitToTikz(std::string circuit_name);
} // namespace ai_pass_selector

#endif // CONVERSION_WORKFLOW_HPP