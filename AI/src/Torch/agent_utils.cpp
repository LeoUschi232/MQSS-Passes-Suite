#include "Torch/agent_utils.hpp"

#include <mlir_utils.hpp>
#include <Environment/environment.hpp>
#include <Passes/CodeGen.hpp>
#include <Torch/parallel_environments.hpp>
#include <Torch/A2C/a2c_ib_fc_lsd.hpp>
#include <Torch/A2C/a2c_ib_fc_lsm.hpp>
#include <Utils/circuit_utils.hpp>
#include <Utils/conversion_workflow.hpp>
#include <Utils/info_utils.hpp>
#include <Utils/passes_utils.hpp>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <torch/torch.h>
#include <llvm/Support/raw_ostream.h>

#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <system_error>
#include <memory>

using namespace mqss::support::quakeDialect;

namespace {

std::string sanitizeKernelName(const std::string &kernel_name) {
  if (kernel_name.empty()) {
    return "kernel";
  }
  std::regex disallowed(R"([-_])");
  std::string sanitized = std::regex_replace(kernel_name, disallowed, "");
  if (sanitized.empty()) {
    return "kernel";
  }
  return sanitized;
}

std::string createEmptyQuakeModule(const std::string &kernel_name,
                                   const std::string &function_name) {
  std::string template_module = R"(module attributes {
  llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  llvm.triple = "x86_64-unknown-linux-gnu",
  quake.mangled_name_map = {__nvqpp__mlirgen__KERNELNAME = "FUNCTIONNAME"}
} {
  func.func @__nvqpp__mlirgen__KERNELNAME() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
   return
  }

  func.func @FUNCTIONNAME(%arg0: !cc.ptr<i8>) {
    return
  }
}
)";
  std::regex kernel_placeholder("KERNELNAME");
  std::regex function_placeholder("FUNCTIONNAME");
  template_module
      = std::regex_replace(template_module, kernel_placeholder, kernel_name);
  template_module = std::regex_replace(
      template_module, function_placeholder, function_name);
  return template_module;
}

std::optional<fs::path> convertQasmCircuitToQuakeFile(
    const fs::path &input_path, std::string &error_message) {
  std::ifstream input_file(input_path);
  if (!input_file.is_open()) {
    error_message = "Failed to open QASM circuit: " + input_path.string();
    return std::nullopt;
  }

  std::stringstream buffer;
  buffer << input_file.rdbuf();
  std::istringstream qasm_stream(buffer.str());

  const std::string kernel_name
      = sanitizeKernelName(input_path.stem().string());
  const std::string function_name = "_" + kernel_name;
  std::string empty_module
      = createEmptyQuakeModule(kernel_name, function_name);

  auto [mlir_module, context_ptr] = extractMLIRContext(empty_module);
  if (context_ptr == nullptr) {
    error_message = "Failed to create MLIR context for QASM conversion.";
    return std::nullopt;
  }
  std::unique_ptr<MLIRContext> context_owner(context_ptr);
  MLIRContext &context = *context_owner;
  context.disableMultithreading();

  mlir::PassManager pass_manager(&context);
  pass_manager.nest<mlir::func::FuncOp>().addPass(
      mqss::opt::createQASM3ToQuakePass(qasm_stream, false));
  pass_manager.addPass(mlir::createCanonicalizerPass());
  pass_manager.addPass(mlir::createCSEPass());
  if (mlir::failed(pass_manager.run(mlir_module))) {
    error_message = "Failed to run QASM to Quake conversion passes for "
        + input_path.string();
    return std::nullopt;
  }

  std::string module_output;
  llvm::raw_string_ostream string_stream(module_output);
  mlir_module.print(string_stream);
  string_stream.flush();

  fs::path temp_model;
  try {
    temp_model = fs::temp_directory_path()
        / fs::path(kernel_name + "-%%%%-%%%%-%%%%-%%%%.qke");
  } catch (const fs::filesystem_error &e) {
    error_message = std::string("Failed to obtain temporary directory: ")
        + e.what();
    return std::nullopt;
  }

  fs::path temp_path;
  try {
    temp_path = fs::unique_path(temp_model);
  } catch (const fs::filesystem_error &e) {
    error_message = std::string("Failed to create temporary file for QASM "
                                "conversion: ")
        + e.what();
    return std::nullopt;
  }

  std::ofstream output_file(temp_path);
  if (!output_file.is_open()) {
    error_message = "Failed to open temporary file for converted circuit: "
        + temp_path.string();
    return std::nullopt;
  }
  output_file << module_output;
  output_file.close();

  return temp_path;
}

} // namespace

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
  fs::path input_path = path / (name + extension);
  struct TemporaryFileGuard {
    std::optional<fs::path> path;
    ~TemporaryFileGuard() {
      if (!path.has_value()) {
        return;
      }
      std::error_code ec;
      fs::remove(*path, ec);
      if (ec) {
        std::cerr << "Warning: failed to clean up temporary file "
            << path->string() << ": " << ec.message() << std::endl;
      }
    }
  } temp_file_guard;

  if (extension == ".qasm") {
    std::string conversion_error;
    if (auto converted_path
        = convertQasmCircuitToQuakeFile(input_path, conversion_error);
        converted_path.has_value()) {
      temp_file_guard.path = converted_path;
      input_path = *converted_path;
    } else {
      std::cerr << conversion_error << std::endl;
      return {};
    }
  } else if (extension == ".qke" || extension == ".quake") {
    // Supported extensions, no additional action required.
  } else {
    std::cerr << "Unsupported circuit extension for "
        << path / (name + extension)
        << ". Supported extensions are .qke, .quake, and .qasm." << std::endl;
    return {};
  }
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