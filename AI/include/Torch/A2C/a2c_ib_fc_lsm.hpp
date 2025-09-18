#ifndef A2C_IB_FC_LSM_HPP
#define A2C_IB_FC_LSM_HPP

#include "Torch/A2C/base_a2c_agent.hpp"

#include "Torch/agent_utils.hpp"

namespace ai_pass_selector {
    /// IB = Instruction Based
    /// FC = Fully Connected
    /// LSM = Layer Size Maintaining
    class A2C_IB_FC_LSM final : public BaseA2CAgent {
    public:
        A2C_IB_FC_LSM(
            int circuit_size_class,
            std::unordered_map<std::string, std::string> params);


        A2C_IB_FC_LSM(
            const std::string &circuit_size,
            std::unordered_map<std::string, std::string> params);

        std::string agentName() const override;
    };
} // namespace ai_pass_selector

#endif // A2C_IB_FC_LSM_HPP
