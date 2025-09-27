#ifndef RANDOMTEST_HPP
#define RANDOMTEST_HPP

// Stdandard library includes
#include <filesystem>

namespace fs = std::filesystem;

namespace ai_pass_selector {
/**
 * @param nr_circuits
 */
void createAndConvertRandomCircuitsToTikz(unsigned int nr_circuits = 20);
/**
 *
 * @param index
 * @return
 */
int createAndConvertOneRandomCircuitToTikz(int index);

} // namespace ai_pass_selector

#endif // RANDOMTEST_HPP