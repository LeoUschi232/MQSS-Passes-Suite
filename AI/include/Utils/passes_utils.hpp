#ifndef PASSES_UTILS_HPP
#define PASSES_UTILS_HPP

// Passes includes
#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Passes/Decompositions.hpp"
#include "Passes/Transforms.hpp"

// Cudaq includes
#include "cudaq/Optimizer/Transforms/Passes.h"

// Mlir includes
#include "mlir/Pass/Pass.h"

// Standard library includes
#include <memory>

using namespace mqss::opt;
using namespace cudaq::opt;

namespace ai_pass_selector {
const std::vector<std::function<std::unique_ptr<BaseMQSSPass>()>>
    PASS_FUNCTIONS = {
        // MQSS Custom passes
        [] { return createZeroRxToIdPass(); },        // 1
        [] { return createZeroRyToIdPass(); },        // 2
        [] { return createZeroRzToIdPass(); },        // 3
        [] { return createCxCxToIdPass(); },          // 4
        [] { return createCyCyToIdPass(); },          // 5
        [] { return createCzCzToIdPass(); },          // 6
        [] { return createXXToIdPass(); },            // 7
        [] { return createYYToIdPass(); },            // 8
        [] { return createZZToIdPass(); },            // 9
        [] { return createSSdgToIdPass(); },          // 10
        [] { return createSdgSToIdPass(); },          // 11
        [] { return createTTdgToIdPass(); },          // 12
        [] { return createTdgTToIdPass(); },          // 13
        [] { return createHHToIdPass(); },            // 14
        [] { return createRxRxToRxPass(); },          // 15
        [] { return createRyRyToRyPass(); },          // 16
        [] { return createRzRzToRzPass(); },          // 17
        [] { return createHXHToZPass(); },            // 18
        [] { return createHZHToXPass(); },            // 19
        [] { return createXHZToHPass(); },            // 20
        [] { return createZHXToHPass(); },            // 21
        [] { return createHRxHToRzPass(); },          // 22
        [] { return createHRzHToRxPass(); },          // 23
        [] { return createHCxHToCzPass(); },          // 24
        [] { return createHCzHToCxPass(); },          // 25
        [] { return createHCrxHToCrzPass(); },        // 26
        [] { return createHCrzHToCrxPass(); },        // 27
        [] { return createXHToHZPass(); },            // 28
        [] { return createHXToZHPass(); },            // 29
        [] { return createYHToHYPass(); },            // 30
        [] { return createHYToYHPass(); },            // 31
        [] { return createZHToHXPass(); },            // 32
        [] { return createHZToXHPass(); },            // 33
        [] { return createSSSToSdgPass(); },          // 34
        [] { return createSdgSdgSdgToSPass(); },      // 35
        [] { return createSSToZPass(); },             // 36
        [] { return createSdgSdgToZPass(); },         // 37
        [] { return createSZToSdgPass(); },           // 38
        [] { return createZSToSdgPass(); },           // 39
        [] { return createSdgZToSPass(); },           // 40
        [] { return createZSdgToSPass(); },           // 41
        [] { return createTTToSPass(); },             // 42
        [] { return createCxCxCxToSwapPass(); },      // 43
        [] { return createCxZToZCxPass(); },          // 44
        [] { return createZCxToCxZPass(); },          // 45
        [] { return createCxXToXCxPass(); },          // 46
        [] { return createXCxToCxXPass(); },          // 47
        [] { return createCxRxToRxCxPass(); },        // 48
        [] { return createRxCxToCxRxPass(); },        // 49
        [] { return createReverseCxPass(); },         // 50
        [] { return createXToHZHPass(); },            // 51
        [] { return createZToHXHPass(); },            // 52
        [] { return createRxToHRzHPass(); },          // 53
        [] { return createRzToHRxHPass(); },          // 54
        [] { return createCxToUpperHCzHPass(); },     // 55
        [] { return createCxToLowerHCzHPass(); },     // 56
        [] { return createCzToUpperHCxHPass(); },     // 57
        [] { return createCzToLowerHCxHPass(); },     // 58
        [] { return createCrxToHCrzHPass(); },        // 59
        [] { return createCrzToHCrxHPass(); },        // 60
        [] { return createSdgToSSSPass(); },          // 61
        [] { return createSToSdgSdgSdgPass(); },      // 62
        [] { return createSToTTPass(); },             // 63
        [] { return createSwapToLowerCxCxCxPass(); }, // 64
        [] { return createSwapToUpperCxCxCxPass(); }, // 65
};
const unsigned int NR_PASSES = PASS_FUNCTIONS.size();

/**
 *
 * @param index
 * @return
 */
std::pair<std::string, std::unique_ptr<mlir::Pass>>
getPassNameAndPointer(unsigned int index);
} // namespace ai_pass_selector

#endif // PASSES_UTILS_HPP
