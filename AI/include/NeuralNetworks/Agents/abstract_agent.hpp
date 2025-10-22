#ifndef ABSTRACT_AGENT_HPP
#define ABSTRACT_AGENT_HPP

// Environment includes
#include "Environment/quantum_circuit.hpp"

// Torch includes
#include "torch/torch.h"

// Mlir includes
#include "mlir/Pass/Pass.h"

// Standard library includes
#include <memory>
#include <mutex>
#include <tuple>

namespace fs = std::filesystem;

namespace ai_pass_selector {

enum class OptimizerType : int;

class AbstractAgent : public torch::nn::Module {
protected:
  /// Attributes on configuration
  unsigned int max_qubits = 0u;
  unsigned int nr_trainable_parmaeters = 0u;

  /// Mutex for thread safety
  std::unique_ptr<std::mutex> model_mutex = std::make_unique<std::mutex>();

public:
  /// Constructors
  explicit AbstractAgent(unsigned int max_qubits);

  /// Destructor
  ~AbstractAgent() override = default;

  /// Copy and move constructors and assignment operators
  AbstractAgent(const AbstractAgent &other) noexcept = delete;

  AbstractAgent(AbstractAgent &&other) noexcept = default;

  AbstractAgent &operator=(const AbstractAgent &other) noexcept = delete;

  AbstractAgent &operator=(AbstractAgent &&other) noexcept = default;

  /// Getters
  unsigned int getMaxQubits() const;

  unsigned int getNrTrainableParameters() const;

  static std::unique_ptr<AbstractAgent> getAgent(const std::string &agent_name);

  /**
   *
   * @param circuit_path
   * @return
   */
  virtual std::vector<std::function<std::unique_ptr<mlir::Pass>()>>
  select_passes_for_circuit(const fs::path &circuit_path) = 0;

  /**
   *
   * @param circuit_path
   * @param pass_functions
   * @return [optimized_circuit, nr_gates_reduction, depth_reduction]
   */
  std::tuple<QuantumCircuit, int, int>
  run_on_circuit(const fs::path &circuit_path,
                 const std::vector<std::function<std::unique_ptr<mlir::Pass>()>>
                     &pass_functions);

  /// Other
  virtual void load_model() = 0;
  virtual std::string agentName() const = 0;
};
} // namespace ai_pass_selector

#endif // ABSTRACT_AGENT_HPP
