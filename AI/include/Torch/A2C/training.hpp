#ifndef TRAINING_HPP
#define TRAINING_HPP

#include <torch/torch.h>

#include <unordered_map>
#include <string>

#include "Torch/A2C/instruction_based_a2c.hpp"

namespace ai_pass_selector {
    std::unordered_map<std::string, std::string> train(
        const InstructionBasedA2CAgent &agent,
        std::string dataset,
        unsigned int episodes,
        double discount_factor = 1.0,
        double gae_hyperparameter = 0.96,
        double entropy_coefficient = 0.01,
        unsigned int max_steps_per_episode = 20,
        torch::Device device = torch::kCPU);
} // namespace ai_pass_selector


#endif // TRAINING_HPP
