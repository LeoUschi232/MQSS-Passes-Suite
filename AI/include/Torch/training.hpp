#ifndef TRAINING_HPP
#define TRAINING_HPP

#include <torch/torch.h>

#include <unordered_map>
#include <string>

#include "agent_utils.hpp"


namespace ai_pass_selector {
    /**
     *
     * @param agent_name
     * @param dataset
     * @param training_params
     * @return
     */
    std::unordered_map<std::string, std::string> train_agent(
        const std::string &agent_name,
        const std::string &dataset,
        std::unordered_map<std::string, std::string> training_params);



} // namespace ai_pass_selector


#endif // TRAINING_HPP
