#include "Torch/agent_utils.hpp"

#include <mlir_utils.hpp>
#include <Environment/environment.hpp>
#include <Torch/parallel_environments.hpp>
#include <Torch/A2C/a2c_ib_fc_lsd.hpp>
#include <Torch/A2C/a2c_ib_fc_lsm.hpp>
#include <Utils/circuit_utils.hpp>
#include <Utils/conversion_workflow.hpp>
#include <Utils/info_utils.hpp>
#include <Utils/passes_utils.hpp>
#include <mlir/Transforms/Passes.h>
#include <torch/torch.h>

using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {

int mapToOptimizerType(std::string &optimizer_name) {
  std::transform(
      optimizer_name.begin(), optimizer_name.end(), optimizer_name.begin(),
      [](unsigned char c) { return std::tolower(c); });
  if (optimizer_name == "adagrad") {
    return OPTIMIZER_ADAGRAD;
  }
  if (optimizer_name == "adam") {
    return OPTIMIZER_ADAM;
  }
  if (optimizer_name == "adamw") {
    return OPTIMIZER_ADAMW;
  }
  if (optimizer_name == "lbfgs") {
    return OPTIMIZER_LBFGS;
  }
  if (optimizer_name == "rmsprop") {
    return OPTIMIZER_RMSPROP;
  }
  if (optimizer_name == "sgd") {
    return OPTIMIZER_SGD;
  }
  // Default to Adam if unknown
  return OPTIMIZER_ADAM;
}

std::unique_ptr<torch::optim::Optimizer> makeOptimizer(
    int optimizerType, const torch::nn::Sequential &agentModel,
    double learningRate) {
  switch (optimizerType) {
  case OPTIMIZER_ADAGRAD:
    return std::make_unique<torch::optim::Adagrad>(
        agentModel->parameters(), torch::optim::AdagradOptions(learningRate));
  case OPTIMIZER_ADAM:
    return std::make_unique<torch::optim::Adam>(
        agentModel->parameters(), torch::optim::AdamOptions(learningRate));
  case OPTIMIZER_ADAMW:
    return std::make_unique<torch::optim::AdamW>(
        agentModel->parameters(), torch::optim::AdamWOptions(learningRate));
  case OPTIMIZER_LBFGS:
    return std::make_unique<torch::optim::LBFGS>(
        agentModel->parameters(), torch::optim::LBFGSOptions(learningRate));
  case OPTIMIZER_RMSPROP:
    return std::make_unique<torch::optim::RMSprop>(
        agentModel->parameters(), torch::optim::RMSpropOptions(learningRate));
  case OPTIMIZER_SGD:
    return std::make_unique<torch::optim::SGD>(
        agentModel->parameters(), torch::optim::SGDOptions(learningRate));
  default:
    throw std::runtime_error(
        "Unsupported optimizer type: " + std::to_string(optimizerType));
  }
}

std::string select_best_agent(const std::string &circuit) {
  auto [module, contex_ptr] = extractMLIRContext(circuit);
  switch (auto [nrQubits, nrGates, depth]
        = getQubitsInstructionsDepth(FuncOp(module));
    classify_circuit(nrQubits, nrGates, depth)) {
  case TINY:
    return "a2c-ib-fc-lsd-5x25x10";
  case SMALL:
    return "a2c-ib-fc-lsd-20x500x50";
  case MODERATE:
    return "a2c-ib-fc-lsd-100x1000x200";
  case BIG:
  case HUGE:
  default:
    break;
  }
  throw std::runtime_error("No suitable agent found for circuit.");
}


std::tuple<std::vector<std::string>, std::vector<unsigned int> >
getRecommendedPasses(
    const std::string &agent_name, const std::string &circuit,
    unsigned int nr_passes, fs::path output_path) {
  auto found_circuit = search_circuit(circuit);
  if (!found_circuit.has_value()) {
    std::cerr << "Failed to get circuit: " << circuit << std::endl;
    return {};
  }
  auto [path, name, extension] = found_circuit.value();
  auto input_path_optional
      = prepare_circuit_input_path(path, name, extension);
  if (!input_path_optional.has_value()) {
    return {};
  }
  fs::path input_path = input_path_optional.value();
  std::vector<std::unique_ptr<mlir::Pass> > passes;
  std::vector<std::string> pass_names;
  std::vector<unsigned int> pass_indexes;

  std::vector<std::string> agent_attributes = split_string(agent_name, '-');
  std::vector<std::string> dimensions
      = split_string(agent_attributes[4], 'x');
  unsigned int max_qubits = std::stoi(dimensions[0]);
  unsigned int max_instructions = std::stoi(dimensions[1]);
  unsigned int max_depth = std::stoi(dimensions[2]);
  constexpr unsigned int NR_ENVS = 1;

  ParallelEnvironments environments(
      NR_ENVS, max_qubits, max_instructions, max_depth,
      nr_passes);

  if (agent_attributes[0] == "a2c") {
    if (agent_attributes[1] == "ib") {
      if (agent_attributes[2] == "fc") {
        if (agent_attributes[3] == "lsd") {
          A2C_IB_FC_LSD agent(max_qubits, max_instructions, max_depth);
          agent.setNrParallelEnvironments(NR_ENVS);

          environments.register_quantum_circuit(0, input_path);
          for (unsigned int pass = 0; pass < nr_passes; pass++) {
            torch::Tensor batched_observations =
                environments.get_batched_instruction_based_observations();
            auto [actions, log_action_probs, state_values, step_entropy]
                = agent.select_action(batched_observations);
            auto [rewards, terminates] = environments.step(actions);
            auto [name, pass_pointer] = getPassNameAndPointer(actions[0]);
            if (pass_pointer == nullptr) {
              break;
            }
            passes.push_back(std::move(pass_pointer));
            pass_names.push_back(name);
            pass_indexes.push_back(actions[0]);
          }

        } else if (agent_attributes[3] == "lsm") {
          A2C_IB_FC_LSM agent(max_qubits, max_instructions, max_depth);
          agent.setNrParallelEnvironments(NR_ENVS);

          environments.register_quantum_circuit(0, input_path);
          for (unsigned int pass = 0; pass < nr_passes; pass++) {
            torch::Tensor batched_observations =
                environments.get_batched_instruction_based_observations();
            auto [actions, log_action_probs, state_values, step_entropy]
                = agent.select_action(batched_observations);
            auto [rewards, terminates] = environments.step(actions);
            auto [name, pass_pointer] = getPassNameAndPointer(actions[0]);
            if (pass_pointer == nullptr) {
              break;
            }
            passes.push_back(std::move(pass_pointer));
            pass_names.push_back(name);
            pass_indexes.push_back(actions[0]);
          }
        }
      }
    }
  }
  if (!output_path.empty()) {
    std::string quake_module_text =
        readFileToString(input_path.string());
    auto [mlir_module, context_ptr] = extractMLIRContext(quake_module_text);
    MLIRContext &context = *context_ptr;
    mlir::PassManager pass_manager(&context);
    for (auto &pass : passes) {
      pass_manager.addPass(std::move(pass));
    }
    pass_manager.addPass(mlir::createCanonicalizerPass());
    pass_manager.addPass(mlir::createCSEPass());
    if (mlir::failed(pass_manager.run(mlir_module))) {
      std::cerr << "\nThe pass failed for " << input_path.string()
          << std::endl;
      return {pass_names, pass_indexes};
    }
    if (int rc = write_module_to_file(mlir_module, output_path);
      rc != 0) {
      return {pass_names, pass_indexes};
    }
  }
  return {pass_names, pass_indexes};
}
} // namespace ai_pass_selector