#include "NeuralNetworks/Agents/training_and_run_manager.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Agents includes
#include "NeuralNetworks/Agents/A2C/a2c_trainer.hpp"
#include "NeuralNetworks/Agents/A2C/base_a2c_agent.hpp"
#include "NeuralNetworks/Agents/ACER/acer_trainer.hpp"
#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"
#include "NeuralNetworks/Agents/PPO/base_ppo_agent.hpp"
#include "NeuralNetworks/Agents/PPO/ppo_trainer.hpp"
#include "NeuralNetworks/Agents/SDSAC/base_sdsac_agent.hpp"
#include "NeuralNetworks/Agents/SDSAC/sdsac_trainer.hpp"
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Utils includes
#include "Utils/dataset_conversion.hpp"
#include "Utils/passes_utils.hpp"
#include "Utils/progress_bar.hpp"

// Stdandard library includes
#include <filesystem>
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
    case AgentClass::A2C: {
      std::unique_ptr<BaseA2CAgent> agent(
          dynamic_cast<BaseA2CAgent *>(abstract_agent.release()));
      if (!agent) {
        throw std::runtime_error("Failed to cast to BaseA2CAgent");
      }
      agent->load_model();
      training_results = train_a2c(agent, dataset);
      break;
    }
    case AgentClass::PPO: {
      std::unique_ptr<BasePPOAgent> agent(
          dynamic_cast<BasePPOAgent *>(abstract_agent.release()));
      if (!agent) {
        throw std::runtime_error("Failed to cast to BasePPOAgent");
      }
      agent->load_model();
      training_results = train_ppo(agent, dataset);
      break;
    }
    case AgentClass::SDSAC: {
      std::unique_ptr<BaseSDSACAgent> agent(
          dynamic_cast<BaseSDSACAgent *>(abstract_agent.release()));
      if (!agent) {
        throw std::runtime_error("Failed to cast to BaseSDSACAgent");
      }
      agent->load_model();
      training_results = train_sdsac(agent, dataset);
      break;
    }
    case AgentClass::ACER: {
      std::unique_ptr<BaseACERAgent> agent(
          dynamic_cast<BaseACERAgent *>(abstract_agent.release()));
      if (!agent) {
        throw std::runtime_error("Failed to cast to BaseACERAgent");
      }
      agent->load_model();
      training_results = train_acer(agent, dataset);
      break;
    }
    default:
      std::cerr << "No such agent yet: " << agent_name << std::endl;
      return {};
    }
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return {};
  }
  return training_results;
}

std::unordered_map<std::string, std::string> run(const std::string &agent_name,
                                                 fs::path circuit_path,
                                                 fs::path output_path) {
  circuit_path = search_circuit(circuit_path).value_or(fs::path());
  if (circuit_path.empty()) {
    return {{"result", "circuit_not_found"}};
  }
  std::unique_ptr<AbstractAgent> agent = AbstractAgent::getAgent(agent_name);
  if (!agent) {
    return {{"result", "agent_not_found"}};
  }
  std::string circuit_name = circuit_path.stem().string();
  std::cout << "Running selection of passes on circuit: " << circuit_name
            << "\nUsing agent: " << agent_name << std::endl;
  std::vector<std::function<std::unique_ptr<Pass>()>> pass_functions =
      agent->select_passes_for_circuit(circuit_path);
  auto [output_circuit, nr_gates_reduction, depth_reduction, pass_names] =
      agent->run_on_circuit(circuit_path, pass_functions);
  std::cout << "Recommended passes:" << std::endl;
  for (const auto &pass_name : pass_names) {
    std::cout << "  " << pass_name << std::endl;
  }
  std::cout << "Gates reduction: " << nr_gates_reduction
            << "\nDepth reduction: " << depth_reduction << std::endl;
  if (output_path.empty()) {
    output_path =
        circuit_path.parent_path() / (circuit_name + "_optimized.qke");
  }
  if (int rc = write_to_file(&output_circuit, output_path); rc != 0) {
    std::cout << "Failed to write optimized circuit to file: " << output_path
              << std::endl;
    return {{"result", "failed"}};
  }
  std::cout << "Optimized circuit written to: " << output_path << std::endl;
  return {{"result", "success"}};
}

std::unordered_map<std::string, std::string>
evaluate(const std::string &agent_name, const std::string &dataset_name,
         std::optional<unsigned int> max_circuits) {
  std::vector<fs::path> files = get_dataset_files(dataset_name);
  if (files.empty()) {
    return {};
  }
  unsigned int nr_files = files.size();
  bool sampled = max_circuits.has_value() && 0u < max_circuits.value() &&
                 max_circuits.value() < nr_files;
  if (sampled) {
    unsigned int nr_sampled_files = max_circuits.value();
    std::uniform_int_distribution distribution(0u, nr_files - 1);
    std::unordered_set<unsigned int> sampled_indices;
    std::vector<fs::path> sampled_files;
    sampled_files.reserve(nr_sampled_files);
    for (unsigned int i = 0u; i < nr_sampled_files; i++) {
      unsigned int idx = distribution(qc_rng());
      while (sampled_indices.find(idx) != sampled_indices.end()) {
        idx = (idx + 1) % nr_files;
      }
      sampled_indices.insert(idx);
      sampled_files.push_back(files[idx]);
    }
    nr_files = nr_sampled_files;
    files = sampled_files;
  }
  std::cout << "Evaluating agent " << agent_name << " on dataset "
            << dataset_name << " with " << nr_files << " circuits."
            << std::endl;
  std::unique_ptr<AbstractAgent> agent = AbstractAgent::getAgent(agent_name);
  if (!agent) {
    std::cerr << "Failed to construct agent " << agent_name << "." << std::endl;
    return {};
  }
  nlohmann::ordered_json optimizations = nlohmann::ordered_json::object();
  double average_nr_gates_reduction = 0.0;
  double average_depth_reduction = 0.0;
  unsigned int progress = 0u;
  for (auto circuit_path : files) {
    std::string circuit_name = circuit_path.stem().string();
    updateProgress(++progress, nr_files,
                   "Evaluating " + agent_name + " on " + circuit_name);

    QuantumCircuitEnvironment qc_environment(agent->getMaxQubits());
    if (!qc_environment.register_quantum_circuit(circuit_path)) {
      std::cerr << "Failed to register quantum circuit: " << circuit_path
                << std::endl;
      return {};
    }
    std::unordered_map<std::string, unsigned int> original_info =
        qc_environment.get_circuit_info();
    optimizations[circuit_name]["nr_qubits"] = original_info["nr_qubits"];
    optimizations[circuit_name]["original_nr_gates"] =
        original_info["nr_gates"];
    optimizations[circuit_name]["original_depth"] = original_info["depth"];
    std::vector<std::function<std::unique_ptr<Pass>()>> pass_functions =
        agent->select_passes_for_circuit(circuit_path);
    auto [quantum_circuit, nr_gates_reduction, depth_reduction, pass_names] =
        agent->run_on_circuit(circuit_path, pass_functions);
    optimizations[circuit_name]["optimized_nr_gates"] =
        quantum_circuit.getNrGates();
    optimizations[circuit_name]["optimized_depth"] = quantum_circuit.getDepth();
    optimizations[circuit_name]["nr_gates_reduction"] = nr_gates_reduction;
    optimizations[circuit_name]["depth_reduction"] = depth_reduction;
    optimizations[circuit_name]["selected_passes"] = pass_names;
    average_nr_gates_reduction += nr_gates_reduction;
    average_depth_reduction += depth_reduction;
  }
  std::cout << std::endl;
  average_nr_gates_reduction /= nr_files;
  average_depth_reduction /= nr_files;
  nlohmann::ordered_json json_file;
  json_file["dataset"] = dataset_name;
  json_file["agent"] = agent_name;
  json_file["average_nr_gates_reduction"] = average_nr_gates_reduction;
  json_file["average_depth_reduction"] = average_depth_reduction;
  json_file["circuit_optimizations"] = optimizations;
  fs::path filepath =
      fs::path(AI_DATASET_DIR) / "Evaluations" /
      ("Evaluation_" + dataset_name + "_" + agent->agentName() + "_" +
       (sampled ? "sample" + std::to_string(nr_files) : "full") + ".json");
  std::ofstream output_stream(filepath);
  output_stream << json_file.dump(4);
  output_stream.close();
  return {{"average_nr_gates_reduction",
           std::to_string(average_nr_gates_reduction)},
          {"average_depth_reduction", std::to_string(average_depth_reduction)}};
}

} // namespace ai_pass_selector