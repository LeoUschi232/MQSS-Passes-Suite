/* This code and any associated documentation is provided "as is"

Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*************************************************************************/
/** @file
 * @brief
 * @details Header file that defines the signature for each MLIR/Quake pass
 * defined in the Munich Quantum Software Stack (MQSS).
 *
 * @par
 * This header must be included to use the collection of cancellation passes
 * that are part of the MQSS.
 */

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/raw_ostream.h"

#include <stdexcept>

namespace mqss::opt {

// Current count:
// 14 passes
std::unique_ptr<mlir::Pass> createXXToIdPass();
std::unique_ptr<mlir::Pass> createYYToIdPass();
std::unique_ptr<mlir::Pass> createZZToIdPass();
std::unique_ptr<mlir::Pass> createSSdgToIdPass();
std::unique_ptr<mlir::Pass> createSdgSToIdPass();
std::unique_ptr<mlir::Pass> createTTdgToIdPass();
std::unique_ptr<mlir::Pass> createTdgTToIdPass();
std::unique_ptr<mlir::Pass> createHHToIdPass();
std::unique_ptr<mlir::Pass> createCxCxToIdPass();
std::unique_ptr<mlir::Pass> createCyCyToIdPass();
std::unique_ptr<mlir::Pass> createCzCzToIdPass();
std::unique_ptr<mlir::Pass> createZeroRxToIdPass();
std::unique_ptr<mlir::Pass> createZeroRyToIdPass();
std::unique_ptr<mlir::Pass> createZeroRzToIdPass();

} // namespace mqss::opt

/**
 * @def GEN_PASS_DECL
 * @brief Macro for declaring passes for registration
 */
// declarative passes
#define GEN_PASS_DECL
/**
 * @def GEN_PASS_REGISTRATION
 * @brief Macro for pass registration
 */
#define GEN_PASS_REGISTRATION
#include "Passes/Cancellations.h.inc"
