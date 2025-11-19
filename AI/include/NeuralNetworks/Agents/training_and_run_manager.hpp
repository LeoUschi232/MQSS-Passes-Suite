#ifndef TRAINING_AND_RUN_MANAGER_HPP
#define TRAINING_AND_RUN_MANAGER_HPP

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

namespace ai_pass_selector {
/**
 * @param agent_name
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train(const std::string &agent_name, const std::string &dataset);

/**
 * @param agent_name
 * @param circuit_path
 * @param output_path
 * @return
 */
std::unordered_map<std::string, std::string>
run(const std::string &agent_name, fs::path circuit_path, fs::path output_path);

/**
 * @param agent_name
 * @param dataset_name
 * @param max_circuits
 * @return
 */
std::unordered_map<std::string, std::string>
evaluate(const std::string &agent_name, const std::string &dataset_name,
         std::optional<unsigned int> max_circuits = std::nullopt);
} // namespace ai_pass_selector
#endif // TRAINING_AND_RUN_MANAGER_HPP
