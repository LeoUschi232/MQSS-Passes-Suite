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
            int circuit_size_class,
            std::unordered_map<std::string, std::string> params);


        A2C_IB_FC_LSD(
            const std::string &circuit_size,
            std::unordered_map<std::string, std::string> params);

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
