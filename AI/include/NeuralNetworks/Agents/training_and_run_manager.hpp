#ifndef TRAINING_AND_RUN_MANAGER_HPP
#define TRAINING_AND_RUN_MANAGER_HPP

#include <optional>
#include <string>
#include <unordered_map>

namespace ai_pass_selector {
/**
 *
 * @param agent_name
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
train(const std::string &agent_name, const std::string &dataset);

/**
 *
 * @param agent_name
 * @param circuit
 * @param output
 * @return
 */
std::unordered_map<std::string, std::string> run(const std::string &agent_name,
                                                 const std::string &circuit,
                                                 const std::string &output);

/**
 *
 * @param agent_name
 * @param dataset_name
 * @return
 */
std::unordered_map<std::string, std::string>
evaluate(const std::string &agent_name, const std::string &dataset_name,
         std::optional<unsigned int> max_circuits = std::nullopt);

} // namespace ai_pass_selector

#endif // TRAINING_AND_RUN_MANAGER_HPP
