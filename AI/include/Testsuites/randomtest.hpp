#ifndef RANDOMTEST_HPP
#define RANDOMTEST_HPP

// Stdandard library includes
#include <filesystem>

namespace fs = std::filesystem;

namespace ai_pass_selector {
/**
 * @param seed
 * @param nr_circuits
 */
void createAndConvertRandomCircuitsToTikz(int seed, unsigned int nr_circuits);

} // namespace ai_pass_selector

#endif // RANDOMTEST_HPP