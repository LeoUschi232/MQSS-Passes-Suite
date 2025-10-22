#ifndef TRAINING_AND_RUN_MANAGER_HPP
#define TRAINING_AND_RUN_MANAGER_HPP

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
 * @param dataset
 * @return
 */
std::unordered_map<std::string, std::string>
evaluate(const std::string &agent_name, const std::string &dataset_name);

} // namespace ai_pass_selector

#endif // TRAINING_AND_RUN_MANAGER_HPP
