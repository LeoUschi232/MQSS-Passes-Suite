#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"

// Neural-Networks includes
#include "NeuralNetworks/Agents/agent_utils.hpp"

// Utils includes
#include "Utils/info_utils.hpp"

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;
BaseACERAgent::BaseACERAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {
  this->acer_retrace_clip_c = GLOBAL_PARAMS["acer_retrace_clip_c"].to_double();
  this->acer_trust_region_delta =
      GLOBAL_PARAMS["acer_trust_region_delta"].to_double();
}

} // namespace ai_pass_selector
