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
*************************************************************************
  author Martin Letras
  date   January 2025
  version 1.0
*************************************************************************/

#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CXCXTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CxCxToId final : public BaseMQSSPass<CxCxToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxCxToId)

  StringRef getArgument() const override { return "CxCxToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive identical Cx gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto cxOp1 = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp1
          || cxOp1.getTargets().size() != 1
          || cxOp1.getControls().size() != 1) {
        return;
      }
      auto optional_cxOp2
          = getNextOperationOnTarget(cxOp1, cxOp1.getTargets()[0]);
      if (!optional_cxOp2) {
        return;
      }
      auto cxOp2 = dyn_cast_or_null<quake::XOp>(*optional_cxOp2);
      if (!cxOp2
          || cxOp2.getTargets().size() != 1
          || cxOp2.getControls().size() != 1
          || cxOp2.getControls()[0] != cxOp1.getControls()[0]) {
        return;
      }
      IRRewriter rewriter(cxOp1->getContext());
      rewriter.eraseOp(cxOp1);
      rewriter.eraseOp(cxOp2);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxCxToIdPass() {
  return std::make_unique<CxCxToId>();
}