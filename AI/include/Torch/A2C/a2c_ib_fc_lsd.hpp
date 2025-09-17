#ifndef A2C_IB_FC_LSD_HPP
#define A2C_IB_FC_LSD_HPP
#include "Torch/A2C/base_a2c_agent.hpp"

#include "Torch/agent_utils.hpp"

namespace ai_pass_selector {
    /// IB = Instruction Based
    /// FC = Fully Connected
    /// LSD = Layer Size Decreasing
    class A2C_IB_FC_LSD final : public BaseA2CAgent {
    public:
        A2C_IB_FC_LSD(
            unsigned int max_qubits,
            unsigned int max_instructions,
            unsigned int max_depth,
            int critic_optimizer_type = OPTIMIZER_ADAM,
            int actor_optimizer_type = OPTIMIZER_ADAM,
            double critic_learning_rate = 0.005,
            double actor_learning_rate = 0.001,
            unsigned int nr_parallel_environments = 10,
            torch::Device device = torch::kCPU);

        std::string agentName() const override;
    };

    /**
     *
     * @param agent
     * @param dataset
     * @param episodes
     * @param discount_factor
     * @param gae_hyperparameter
     * @param entropy_coefficient
     * @param max_steps_per_episode
     * @return
     */
    std::unordered_map<std::string, std::string> train_agent(
        A2C_IB_FC_LSD &agent,
        std::string dataset,
        unsigned int episodes,
        double discount_factor,
        double gae_hyperparameter,
        double entropy_coefficient,
        unsigned int max_steps_per_episode);
} // namespace ai_pass_selector

#endif // A2C_IB_FC_LSD_HPP
