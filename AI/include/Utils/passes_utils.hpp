#ifndef PASSES_UTILS_HPP
#define PASSES_UTILS_HPP

#include "Cancellations.hpp"
#include "Decompositions.hpp"
#include "Transforms.hpp"

#include "mlir/Pass/Pass.h"

// Standard library includes
#include <memory>

using namespace mqss::opt;

namespace ai_pass_selector {
    constexpr unsigned int NR_PASSES = 65;

    constexpr std::array<std::unique_ptr<mlir::Pass>(*)(), NR_PASSES> PASS_FUNCTIONS = {
        // Custom passes
        &createZeroRxToIdPass, // 1
        &createZeroRyToIdPass, // 2
        &createZeroRzToIdPass, // 3
        &createCxCxToIdPass, // 4
        &createCyCyToIdPass, // 5
        &createCzCzToIdPass, // 6
        &createXXToIdPass, // 7
        &createYYToIdPass, // 8
        &createZZToIdPass, // 9
        &createSSdgToIdPass, // 10
        &createSdgSToIdPass, // 11
        &createTTdgToIdPass, // 12
        &createTdgTToIdPass, // 13
        &createHHToIdPass, // 14
        &createRxRxToRxPass, // 15
        &createRyRyToRyPass, // 16
        &createRzRzToRzPass, // 17
        &createHXHToZPass, // 18
        &createHZHToXPass, // 19
        &createXHZToHPass, // 20
        &createZHXToHPass, // 21
        &createHRxHToRzPass, // 22
        &createHRzHToRxPass, // 23
        &createHCxHToCzPass, // 24
        &createHCzHToCxPass, // 25
        &createHCrxHToCrzPass, // 26
        &createHCrzHToCrxPass, // 27
        &createXHToHZPass, // 28
        &createHXToZHPass, // 29
        &createYHToHYPass, // 30
        &createHYToYHPass, // 31
        &createZHToHXPass, // 32
        &createHZToXHPass, // 33
        &createSSSToSdgPass, // 34
        &createSdgSdgSdgToSPass, // 35
        &createSSToZPass, // 36
        &createSdgSdgToZPass, // 37
        &createSZToSdgPass, // 38
        &createZSToSdgPass, // 39
        &createSdgZToSPass, // 40
        &createZSdgToSPass, // 41
        &createTTToSPass, // 42
        &createCxCxCxToSwapPass, // 43
        &createCxZToZCxPass, // 44
        &createZCxToCxZPass, // 45
        &createCxXToXCxPass, // 46
        &createXCxToCxXPass, // 47
        &createCxRxToRxCxPass, // 48
        &createRxCxToCxRxPass, // 49
        &createReverseCxPass, // 50
        &createXToHZHPass, // 51
        &createZToHXHPass, // 52
        &createRxToHRzHPass, // 53
        &createRzToHRxHPass, // 54
        &createCxToUpperHCzHPass, // 55
        &createCxToLowerHCzHPass, // 56
        &createCzToUpperHCxHPass, // 57
        &createCzToLowerHCxHPass, // 58
        &createCrxToHCrzHPass, // 59
        &createCrzToHCrxHPass, // 60
        &createSdgToSSSPass, // 61
        &createSToSdgSdgSdgPass, // 62
        &createSToTTPass, // 63
        &createSwapToLowerCxCxCxPass, // 64
        &createSwapToUpperCxCxCxPass // 65
        // Already existing Cudaq passes

    };


    /**
     *
     * @param index
     * @return
     */
    std::pair<std::string, std::unique_ptr<mlir::Pass> >
    getPassNameAndPointer(unsigned int index);
} // namespace ai_pass_selector

#endif // PASSES_UTILS_HPP
