#include "NeuralNetworks/Agents/SDSAC/sdsac_trainer.hpp"

// Utils includes
#include "Utils/info_utils.hpp"

namespace ai_pass_selector {
extern std::unordered_map<std::string, PassSelectorRuntimeParam>
    GLOBAL_PARAMS;

std::unordered_map<std::string, std::string>
train_sdsac(const std::unique_ptr<BaseSDSACAgent> &agent,
            const std::string &dataset) {
  return {};
}

} // namespace ai_pass_selector