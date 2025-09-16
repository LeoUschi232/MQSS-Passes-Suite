#ifndef PARALLEL_ENVIRONMENTS_HPP
#define PARALLEL_ENVIRONMENTS_HPP
#include <Environment/environment.hpp>

#include <torch/torch.h>

namespace ai_pass_selector {
    class ParallelEnvironments {
        unsigned int nr_environments;
        std::vector<QuantumCircuitEnviorment> environments;
        unsigned int max_qubits;
        unsigned int max_instructions;
        unsigned int max_depth;
        unsigned int max_steps;

    public:
        /// Constructor
        ParallelEnvironments(
            unsigned int nr_environments,
            unsigned int max_qubits,
            unsigned int max_instructions,
            unsigned int max_depth,
            unsigned int max_steps);

        /// Destructor
        ~ParallelEnvironments() = default;

        /// Copy and move constructors and assignment operators
        ParallelEnvironments(const ParallelEnvironments &other) = delete;

        ParallelEnvironments(ParallelEnvironments &&other) noexcept = default;

        ParallelEnvironments &operator=(const ParallelEnvironments &other) = delete;

        ParallelEnvironments &operator=(ParallelEnvironments &&) noexcept = default;

        /**
         *
         * @param index
         * @param circuit_path
         * @return
         */
        bool register_quantum_circuit(
            unsigned int index, const fs::path &circuit_path);

        /**
         *
         * @return
         */
        torch::Tensor get_batched_instruction_based_observations() const;

        /**
         *
         * @return
         */
        torch::Tensor get_batched_depth_based_observations() const;

        /**
         *
         * @return
         */
        unsigned int size() const;


        /**
         *
         */
        void clear_circuits();

        /**
         *
         * @param actions
         * @return
         */
        std::vector<std::tuple<double, bool> > step(
            std::vector<unsigned int> actions);
    };
} // namespace ai_pass_selector

#endif //PARALLEL_ENVIRONMENTS_HPP
