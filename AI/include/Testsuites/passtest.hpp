#ifndef PASSTEST_HPP
#define PASSTEST_HPP

// MLIR includes
#include "mlir/Pass/Pass.h"

// Stdandard library includes
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace ai_pass_selector {
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
} // namespace ai_pass_selector

#endif // PASSTEST_HPP