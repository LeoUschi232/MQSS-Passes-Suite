#include "NeuralNetworks/Agents/training_and_run_manager.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Agents includes
#include "NeuralNetworks/Agents/A3C/a3c_agents.hpp"
#include "NeuralNetworks/Agents/A3C/a3c_trainer.hpp"
#include "NeuralNetworks/Agents/A3C/base_a3c_agent.hpp"
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Utils includes
#include "Utils/passes_utils.hpp"
#include "Utils/progress_bar.hpp"

// Stdandard library includes
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

std::unordered_map<std::string, std::string>
train(const std::string &agent_name, const std::string &dataset) {
  std::unordered_map<std::string, std::string> training_results;
  std::unique_ptr<AbstractAgent> abstract_agent =
      AbstractAgent::getAgent(agent_name);
  try {
    switch (AgentAttributes attributes = parseAgentName(agent_name);
            attributes.agent_class) {
    case AgentClass::A3C: {
      std::unique_ptr<BaseA3CAgent> agent(
          dynamic_cast<BaseA3CAgent *>(abstract_agent.release()));
      if (!agent) {
        throw std::runtime_error("Failed to cast to BaseA3CAgent");
      }
      agent->load_model();
      unsigned int nr_asynchronous_agents =
          GLOBAL_PARAMS["nr_asynchronous_agents"].to_int();
      if (nr_asynchronous_agents <= 1u) {
        std::cout << "Only 1 asnc A3C agent => Defaulting to A2C training."
                  << std::endl;
        training_results = train_a2c(agent, dataset);
      } else {
        training_results = train_a3c(agent, dataset);
      }
      break;
    }
    default:
      std::cerr << "No such agent yet: " << agent_name << std::endl;
      return {};
    }
  } catch (const std::runtime_error &e) {
    std::cerr << "\n" << e.what() << std::endl;
    return {};
  }
  return training_results;
}

std::unordered_map<std::string, std::string> run(const std::string &agent_name,
                                                 const std::string &circuit,
                                                 const std::string &output) {
  throw std::runtime_error("Not implemented yet");
}

std::unordered_map<std::string, std::string>
evaluate(const std::string &agent_name, const std::string &dataset_name) {
  std::vector<fs::path> files = get_dataset_files(dataset_name);
  if (files.empty()) {
    return {};
  }
  unsigned int nr_files = files.size();
  std::cout << "Evaluating agent " << agent_name << " on dataset "
            << dataset_name << " with " << nr_files << " circuits."
            << std::endl;
  std::unique_ptr<AbstractAgent> agent = AbstractAgent::getAgent(agent_name);
  agent->load_model();
  std::vector<std::tuple<std::string, unsigned int, unsigned int>>
      circuit_optimization_results;
  double avg_nr_gates_reduction = 0.0;
  double avg_depth_reduction = 0.0;
  unsigned int progress = 0u;
  for (auto circuit_path : files) {
    updateProgress(++progress, nr_files,
                   "Extracting statistics from: " + dataset_name);
    std::vector<std::function<std::unique_ptr<Pass>()>> pass_functions =
        agent->select_for_circuit(circuit_path);
    auto [_, nr_gates_reduction, depth_reduction] =
        agent->run_on_circuit(circuit_path, pass_functions);
    circuit_optimization_results.emplace_back(
        circuit_path.stem().string(), nr_gates_reduction, depth_reduction);
    avg_nr_gates_reduction += nr_gates_reduction;
    avg_depth_reduction += depth_reduction;
  }
  avg_nr_gates_reduction /= nr_files;
  avg_depth_reduction /= nr_files;
  nlohmann::json json_file;
  json_file["dataset_name"] = dataset_name;
  json_file["agent"] = agent_name;
  nlohmann::json optimizations = nlohmann::json::object();
  for (const auto &[circuit_name, nr_gates_reduction, depth_reduction] :
       circuit_optimization_results) {
    optimizations[circuit_name] = {nr_gates_reduction, depth_reduction};
  }
  json_file["circuit_optimizations"] = optimizations;
  fs::path filepath =
      fs::path(AI_DATASET_DIR) / (dataset_name + "_evaluation.json");
  std::ofstream output_stream(filepath);
  output_stream << json_file.dump(/*ident=*/4);
  output_stream.close();
  return {{"avg_nr_gates_reduction", std::to_string(avg_nr_gates_reduction)},
          {"avg_depth_reduction", std::to_string(avg_depth_reduction)}};
}

} // namespace ai_pass_selector