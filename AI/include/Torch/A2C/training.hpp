#ifndef TRAINING_HPP
#define TRAINING_HPP

#include <torch/torch.h>

#include <unordered_map>
#include <string>

namespace ai_pass_selector {
std::unordered_map<std::string, std::string> train(
    const torch::nn::Module &agent,
    unsigned int episodes,
    double discount_factor = 1.0,
    double gae_hyperparameter = 0.96,
    double entropy_coefficient = 0.01,
    torch::Device device = torch::kCPU);

} // namespace ai_pass_selector


#endif // TRAINING_HPP