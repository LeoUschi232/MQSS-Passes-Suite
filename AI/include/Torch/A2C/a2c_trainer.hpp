#ifndef A2C_TRAINER_HPP
#define A2C_TRAINER_HPP

#include "base_a2c_agent.hpp"

#include <vector>
#include <string>
#include <unordered_map>

namespace ai_pass_selector {
    struct A2CTrainerConfig {
        unsigned int nr_parallel_environments = 10;
        unsigned int episodes = 1000;
        unsigned int max_steps_per_episode = 20;
        double discount_factor = 1.0;
        double gae_hyperparameter = 0.96;
        double entropy_coefficient = 0.01;
        torch::Device device = torch::kCPU;
        std::vector<std::string> applied_overrides;
    };

    A2CTrainerConfig build_a2c_trainer_config(
        const std::unordered_map<std::string, std::string> &params);

    /**
     *
     * @param agent
     * @param dataset
     * @param params
     * @return
     */
    std::unordered_map<std::string, std::string> train_a2c(
        BaseA2CAgent &agent,
        const std::string &dataset,
        std::unordered_map<std::string, std::string> params);
} // namespace ai_pass_selector

#endif // A2C_TRAINER_HPP
