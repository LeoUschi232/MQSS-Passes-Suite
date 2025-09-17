#ifndef INFO_UTILS_HPP
#define INFO_UTILS_HPP

#include <string>
#include <filesystem>
#include <optional>
#include <random>

namespace fs = std::filesystem;

namespace ai_pass_selector {
static std::mt19937 rng(std::random_device{}());

/**
 *
 * @param start
 * @param end
 * @return
 */
int random_int(int start, int end);

/**
 * Given a circuit name or its full filepath, search for the circuit and
 * returns its folder, name and extension.
 * @param circuit_file Name or full filepath of the circuit to search for.
 * @return Optional tuple of 3 values:
 *  - Absolute path to folder where circuit was found as fs::path
 *  - The circuit name without extension as std::string
 *  - The extension of the circuit file as std::string
 */
std::optional<std::tuple<fs::path, std::string, std::string> >
search_circuit(const std::string &circuit_file);

/**
 * Resolve a circuit lookup result into a usable Quake file path.
 * @param circuit_folder Absolute directory that contains the circuit.
 * @param circuit_name   Circuit basename (without extension).
 * @param circuit_extension Original extension returned by search_circuit.
 * @return The full path to a Quake (.qke) file when available.
 */
std::optional<fs::path> prepare_circuit_input_path(
    const fs::path &circuit_folder,
    const std::string &circuit_name,
    const std::string &circuit_extension);

/**
 *
 * @param circuit_file
 * @return
*/
std::optional<std::tuple<
  std::string, std::string,
  unsigned int, unsigned int, unsigned int> >
get_circuit_info(const std::string &circuit_file);

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
std::optional<std::tuple<
  unsigned int,
  unsigned int, double, unsigned int,
  unsigned int, double, unsigned int,
  unsigned int, double, unsigned int> >
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

/**
 *
 * @param dataset_name
 * @return
 */
std::vector<fs::path> get_dataset_files(const std::string &dataset_name);


} // namespace ai_pass_selector

#endif // INFO_UTILS_HPP