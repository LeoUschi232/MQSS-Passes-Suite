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
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_ZCXTOCXZ

#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

void commuteZCx(Operation *currentOp) {
  auto currentGate = dyn_cast_or_null<quake::XOp>(*currentOp);
  if (!currentGate || currentGate.getControls().size() != 1 ||
      currentGate.getTargets().size() != 1) {
    return;
  }
  auto prevOp = supportQuake::getPreviousOperationOnTarget(
      currentGate, currentGate.getControls()[0]);
  if (!prevOp) {
    return;
  }
  auto previousGate = dyn_cast_or_null<quake::ZOp>(prevOp);
  if (!previousGate || previousGate.getControls().size() != 0 ||
      previousGate.getTargets().size() != 1) {
    return;
  }
  int targetPrev = supportQuake::extractIndexFromQuakeExtractRefOp(
      previousGate.getTargets()[0].getDefiningOp());
  int controlCurr = supportQuake::extractIndexFromQuakeExtractRefOp(
      currentGate.getControls()[0].getDefiningOp());
  if (targetPrev == controlCurr) {
    IRRewriter rewriter(currentGate->getContext());
    rewriter.setInsertionPointAfter(currentGate);
    rewriter.create<quake::ZOp>(previousGate.getLoc(), previousGate.isAdj(),
                                previousGate.getParameters(),
                                previousGate.getControls(),
                                previousGate.getTargets());
    rewriter.eraseOp(previousGate);
    return;
  }
}

class ZCxToCxZ final : public BaseMQSSPass<ZCxToCxZ> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZCxToCxZ)

  StringRef getArgument() const override { return "ZCxToCxZ"; }

  StringRef getDescription() const override {
    return "Apply commutation pass to pattern Z-CNot to CNot-Z";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) { commuteZCx(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZCxToCxZPass() {
  return std::make_unique<ZCxToCxZ>();
}
