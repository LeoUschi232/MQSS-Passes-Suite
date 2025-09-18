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
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_REVERSECX

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

class ReverseCx final : public BaseMQSSPass<ReverseCx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReverseCx)

  StringRef getArgument() const override { return "ReverseCx"; }

  StringRef getDescription() const override {
    return "Transforms CX(0,1) to H(0) H(1) CX(1,0) H(1) H(0)";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto cxOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp
          || cxOp.getControls().size() != 1
          || cxOp.getTargets().size() != 1) {
        return;
      }
      Value control = cxOp.getControls()[0];
      Value target = cxOp.getTargets()[0];
      Location loc = cxOp.getLoc();

      IRRewriter rewriter(cxOp->getContext());
      rewriter.setInsertionPointAfter(cxOp);
      rewriter.create<quake::HOp>(loc, control);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.create<quake::XOp>(loc, target, control);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.create<quake::HOp>(loc, control);
      rewriter.eraseOp(cxOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createReverseCxPass() {
  return std::make_unique<ReverseCx>();
}