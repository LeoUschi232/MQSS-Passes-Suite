#ifndef INFO_UTILS_HPP
#define INFO_UTILS_HPP

// Support includes
#include "Interfaces/Constants.hpp"

// Standard Library includes
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace ai_pass_selector {
////////////////////////////////////////////////////////////////////////////////
/// Mersenne Twister RNG
inline std::mt19937 &qc_rng() {
  static std::mt19937 rng_engine{std::random_device{}()};
  return rng_engine;
}
inline void seed_qc_rng(uint32_t seed) { qc_rng().seed(seed); }
inline double random01() {
  thread_local std::uniform_real_distribution dist01(0.0, 1.0);
  return dist01(qc_rng());
}
inline double randomAngle() {
  thread_local std::uniform_real_distribution distAngle(0.0, 2.0 * PI);
  return distAngle(qc_rng());
}
inline int randomInt(int start, int end) {
  std::uniform_int_distribution distInt(start, end - 1);
  return distInt(qc_rng());
}
////////////////////////////////////////////////////////////////////////////////

/**
 *
 * @param str
 * @param delimiter
 * @return
 */
std::vector<std::string> split_string(const std::string &str, char delimiter);

/**
 * Given a circuit name or its full filepath, search for the circuit and
 * returns its folder, name and extension.
 * @param circuit Name or full filepath of the circuit to search for.
 * @return
 */
std::optional<fs::path> search_circuit(const std::string &circuit);

/**
 *
 * @param circuit
 * @return
 */
std::optional<std::tuple<fs::path, unsigned int, unsigned int, unsigned int>>
get_circuit_info(const std::string &circuit);

/**
 *
 * @param circuit_file
 */
void print_circuit_info(const std::string &circuit_file);

/**
 *
 * @param dataset_name
 * @return
 */
std::vector<fs::path> get_dataset_files(const std::string &dataset_name);

/**
 *
 * @param dataset_name
 * @return
 */
std::optional<std::vector<std::pair<std::string, std::string>>>
get_dataset_info(const std::string &dataset_name);

/**
 *
 * @param dataset_name
 */
void print_dataset_info(const std::string &dataset_name);

/**
 *
 * @param agent_name
 */
void print_agent_info(const std::string &agent_name);
} // namespace ai_pass_selector

#endif // INFO_UTILS_HPP
