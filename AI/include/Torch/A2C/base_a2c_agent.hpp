#ifndef BASE_A2C_AGENT_HPP
#define BASE_A2C_AGENT_HPP

// Torch includes
#include <torch/torch.h>

// Standard library includes
#include <utility>
#include <tuple>
#include <memory>
#include <filesystem>


namespace fs = std::filesystem;

namespace ai_pass_selector {
    class BaseA2CAgent : public torch::nn::Module {
    protected:
        const unsigned int max_qubits;
        const unsigned int max_instructions;
        const unsigned int max_depth;
        const int critic_optimizer_type;
        const int actor_optimizer_type;
        const double critic_learning_rate;
        const double actor_learning_rate;
        const unsigned int nr_parallel_environments;
        torch::Device device;
        unsigned int nr_input_values;
        unsigned int nr_output_values;
        torch::nn::Sequential critic;
        torch::nn::Sequential actor;
        std::unique_ptr<torch::optim::Optimizer> actor_optimizer;
        std::unique_ptr<torch::optim::Optimizer> critic_optimizer;

    public:
        /// Constructor
        BaseA2CAgent(
            unsigned int max_qubits,
            unsigned int max_instructions,
            unsigned int max_depth,
            int critic_optimizer_type,
            int actor_optimizer_type,
            double critic_learning_rate,
            double actor_learning_rate,
            unsigned int nr_parallel_environments,
            torch::Device device)
            : max_qubits(max_qubits),
              max_instructions(max_instructions),
              max_depth(max_depth),
              critic_optimizer_type(critic_optimizer_type),
              actor_optimizer_type(actor_optimizer_type),
              critic_learning_rate(critic_learning_rate),
              actor_learning_rate(actor_learning_rate),
              nr_parallel_environments(
                  nr_parallel_environments),
              device(device),
              nr_input_values(0),
              nr_output_values(0) {
        }

        /**
         *
         * @param nr_input_values
         * @param nr_output_values
         * @param critic
         * @param actor
         * @return
         */
        bool initialize(
            int nr_input_values,
            int nr_output_values,
            const torch::nn::Sequential &critic,
            const torch::nn::Sequential &actor);

        /// Destructor
        ~BaseA2CAgent() override = default;

        /// Copy and move constructors and assignment operators
        BaseA2CAgent(const BaseA2CAgent &other) = delete;

        BaseA2CAgent(BaseA2CAgent &&other) noexcept = default;

        BaseA2CAgent &operator=(const BaseA2CAgent &other) = delete;

        BaseA2CAgent &operator=(BaseA2CAgent &&other) noexcept = delete;

        unsigned int getNrParallelEnvironments() const;

        torch::Device getDevice() const;

        unsigned int getNrActions() const;

        /**
         *
         * @param batched_observations
         * @return
         */
        std::pair<torch::Tensor, torch::Tensor> forward(
            torch::Tensor batched_observations);

        /**
         *
         * @param batched_observations
         * @return
         */
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        select_action(const torch::Tensor &batched_observations);

        /**
         *
         * @param rewards
         * @param log_action_probs
         * @param state_values
         * @param entropy
         * @param termination_masks
         * @param discount_factor
         * @param gae_hyperparameter
         * @param entropy_coefficient
         * @return
         */
        static std::pair<torch::Tensor, torch::Tensor> get_losses(
            const torch::Tensor &rewards,
            const torch::Tensor &log_action_probs,
            const torch::Tensor &state_values,
            const torch::Tensor &entropy,
            const torch::Tensor &termination_masks,
            double discount_factor,
            double gae_hyperparameter,
            double entropy_coefficient);


        /**
         *
         * @param critic_loss
         * @param actor_loss
         */
        void update_parameters(const torch::Tensor &critic_loss,
                               const torch::Tensor &actor_loss) const;

        /**
         *
         */
        void save_model() const;

        /**
         *
         */
        void load_model();

        /**
         *
         * @return
         */
        virtual std::string model_name() const = 0;

        /**
         *
         * @return
         */
        virtual std::unordered_map<std::string, std::string> train();
    };
} // namespace ai_pass_selector

#endif // BASE_A2C_AGENT_HPP
