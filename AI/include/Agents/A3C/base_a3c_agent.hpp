#ifndef BASE_A3C_AGENT_HPP
#define BASE_A3C_AGENT_HPP

// Torch includes
#include "torch/torch.h"

namespace fs = std::filesystem;

namespace ai_pass_selector {

class BaseA3CAgent : public torch::nn::Module {};
} // namespace ai_pass_selector

#endif // BASE_A3C_AGENT_HPP
