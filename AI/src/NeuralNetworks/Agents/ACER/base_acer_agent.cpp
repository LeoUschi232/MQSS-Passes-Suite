#include "NeuralNetworks/Agents/ACER/base_acer_agent.hpp"

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;
BaseACERAgent::BaseACERAgent(unsigned int max_qubits)
    : BaseActorCritic(max_qubits) {
  double ppo_epsilon = GLOBAL_PARAMS["ppo_epsilon"].to_double();
  this->acer_retrace_clip_c = 1.0 - ppo_epsilon;
  this->acer_trust_region_delta = 1.0 + ppo_epsilon;
}



} // namespace ai_pass_selector
