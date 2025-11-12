#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Utils includes
#include "Utils/info_utils.hpp"

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;
BaseACERAgent::BaseACERAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {
  this->acer_truncation_threshold_c =
      GLOBAL_PARAMS["acer_truncation_threshold_c"].to_double();
  this->acer_trust_region_delta =
      GLOBAL_PARAMS["acer_trust_region_delta"].to_double();
}

torch::Tensor BaseACERAgent::get_value(const torch::Tensor &observation) {
  auto [policy, Q_values] = this->forward(
      /*observation=*/observation.to(this->device).to(torch::kFloat32));
  return policy.dot(Q_values).unsqueeze(-1); // Shape []
}
} // namespace ai_pass_selector
