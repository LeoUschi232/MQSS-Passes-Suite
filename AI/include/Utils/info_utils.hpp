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
    std::optional<fs::path>
    search_circuit(const std::string &circuit);

    /**

    /**
     *
     * @param circuit
     * @return
    */
    std::optional<std::tuple<
        fs::path, unsigned int, unsigned int, unsigned int> >
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
} // namespace ai_pass_selector

#endif // INFO_UTILS_HPP
