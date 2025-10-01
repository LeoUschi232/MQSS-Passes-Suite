#ifndef TENSORTEST_HPP
#define TENSORTEST_HPP

// Stdandard library includes
#include <filesystem>

namespace fs = std::filesystem;

namespace ai_pass_selector {
constexpr unsigned int TENSORTEST_MAX_QUBITS = 5;

/**
 * Convert all tensor test circuits in the AI/Dataset/Quake/Tensortest
 * directory to Tensors for AI input representation and then back again to check
 * if the representations are consistent.
 * Creates a pdf with visualized circuit at
 * AI/Dataset/Latex/tensortest_quantum_circuits.pdf
 */
void convertAllTensortestCircuitsToTikz();

/**
 * Convert a single tensor test circuit in the AI/Dataset/Quake/Tensortest
 * directory to Tensors for AI input representation and then back again to
 * check if the representations are consistent.
 * Creates tikz files at
 * AI/Dataset/Latex/Tensortest/tensortest{index}_input.tikz and
 * AI/Dataset/Latex/Tensortest/tensortest{index}_output.tikz
 * @param index Index of the tensortest circuit to be converted.
 * @return 0 on success, -1 on failure.
 */
int convertTensortestCircuitToTikz(int index);
} // namespace ai_pass_selector

#endif // TENSORTEST_HPP