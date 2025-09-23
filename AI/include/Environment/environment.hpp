#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

////////////////////////////////////////////////////////////////////////////////
/// The includes of llvm Casting must be left here before the include of cudaq
/// QuakeOps otherwise the comipler will complain that these operations do not
/// exist in the header file.
#include "Support/mlir_utils.hpp"

#include "llvm/Support/Casting.h"
using llvm::isa;
using llvm::cast;
using llvm::dyn_cast;
////////////////////////////////////////////////////////////////////////////////

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"

// Standard library includes
#include <string>
#include <unordered_map>
#include <filesystem>

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::ModuleOp;
////////////////////////////////////////////////////////////////////////////////

using namespace mqss::support::quakeDialect;
namespace fs = std::filesystem;


namespace ai_pass_selector {
    constexpr double PI = 3.14159265358979323846;
    constexpr double TWO_PI = 6.28318530717958647692;

    constexpr unsigned int CIRCUIT_VALID = 0;
    constexpr unsigned int NO_CIRCUIT = 1;
    constexpr unsigned int TOO_MANY_QUBITS = 2;
    constexpr unsigned int TOO_MANY_INSTRUCTIONS = 3;
    constexpr unsigned int TOO_LARGE_DEPTH = 4;
    constexpr unsigned int NO_QUBIT_ALLOCATIONS = 5;
    constexpr unsigned int MULTIPLE_QUBIT_ALLOCATIONS = 6;
    constexpr unsigned int AMBIGUOUS_MEASUREMENT = 7;

    inline std::vector<double> params_to_angles(std::vector<double> params) {
        for (int i = 0; i < params.size(); i++) {
            double angle = std::fmod(params[i] + PI, TWO_PI);
            if (angle < 0) {
                angle += TWO_PI;
            }
            params[i] = angle - PI;
        }
        return params;
    }

    class QuantumCircuitEnviorment {
        unsigned int max_qubits;
        unsigned int max_instructions;
        unsigned int max_depth;
        fs::path circuit_path;
        ModuleOp circuit_module;
        std::unique_ptr<MLIRContext *> context_ptr;
        unsigned int max_steps;
        unsigned int current_step;

        bool register_quantum_circuit(
            const fs::path &circuit_path, const std::string &circuit_text);

    public:
        /// Constructors
        QuantumCircuitEnviorment(
            unsigned int max_qubits,
            unsigned int max_instructions,
            unsigned int max_depth,
            unsigned int max_steps)
            : max_qubits(max_qubits),
              max_instructions(max_instructions),
              max_depth(max_depth),
              context_ptr(nullptr),
              max_steps(max_steps),
              current_step(0) {
        }

        QuantumCircuitEnviorment(
            unsigned int max_qubits,
            unsigned int max_instructions,
            unsigned int max_depth,
            const fs::path &circuit_path,
            unsigned int max_steps);

        /// Destructor
        ~QuantumCircuitEnviorment() = default;

        /// Copy and move constructors and assignment operators
        QuantumCircuitEnviorment(const QuantumCircuitEnviorment &other) = delete;

        QuantumCircuitEnviorment(QuantumCircuitEnviorment &&other) noexcept = default;

        QuantumCircuitEnviorment
        &operator=(const QuantumCircuitEnviorment &other) = delete;

        QuantumCircuitEnviorment &operator=(QuantumCircuitEnviorment &&) noexcept = default;

        /**
         *
         */
        void clear_circuit();

        /**
         *
         * @param circuit_path
         */
        bool register_quantum_circuit(const fs::path &circuit_path);

        /**
         *
         */
        void reset();

        /**
         *
         * @param circuit
         * @return
         */
        static std::unordered_map<std::string, unsigned int>
        get_circuit_info(FuncOp circuit);

        /**
         *
         * @param circuit
         * @return
         */
        unsigned int circuit_invalid_type(FuncOp circuit) const;


        /**
         *
         * @return
         */
        std::unordered_map<std::string, unsigned int> get_circuit_info() const;

        /**
         *
         * @return
         */
        AllInstructionsTensor<double> get_instruction_based_observation();

        /**
         *
         * @return
         */
        AllDepthsTensor<double> get_depth_based_observation();

        /**
         *
         * @param action
         * @return
         */
        std::tuple<double, bool> step(unsigned int action);

        /**
         *
         * @param op
         * @return
         */
        static
        std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, bool>
        getOperatingControlsTargetsParams(Operation *op);
    };
} // namespace ai_pass_selector

#endif // ENVIRONMENT_HPP
