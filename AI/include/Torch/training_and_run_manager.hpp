#ifndef TRAINING_HPP
#define TRAINING_HPP

#include <unordered_map>
#include <string>


namespace ai_pass_selector {
/**
 *
 * @param agent_name
 * @param dataset
 * @param params
 * @return
 */
std::unordered_map<std::string, std::string> train(
    const std::string &agent_name,
    const std::string &dataset,
    std::unordered_map<std::string, std::string> params);

/**
 *
 * @param agent_name
 * @param circuit
 * @param output
 * @param params
 * @return
 */
std::unordered_map<std::string, std::string> run(
    const std::string &agent_name,
    const std::string &circuit,
    const std::string &output,
    std::unordered_map<std::string, std::string> params);


} // namespace ai_pass_selector


#endif // TRAINING_HPP
