/* This code and any associated documentation is provided "as is"

Copyright 2025 Munich Quantum Software Stack Project

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
  date   February 2025
  version 1.0
*************************************************************************/

#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_SWAPTOLOWERCXCXCX

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

struct ReplaceSwapToLowerCxCxCx final : OpRewritePattern<quake::SwapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      quake::SwapOp swapOp, PatternRewriter &rewriter) const override {
    if (!swapOp.getControls().empty() || swapOp.getTargets().size() != 2) {
      return success();
    }
    Location loc = swapOp.getLoc();
    Value q0 = swapOp.getTargets()[0];
    Value q1 = swapOp.getTargets()[1];
    rewriter.create<quake::XOp>(loc, q1, q0);
    rewriter.create<quake::XOp>(loc, q0, q1);
    rewriter.create<quake::XOp>(loc, q1, q0);
    rewriter.replaceOp(swapOp, {});
    return success();
  }
};

class SwapToLowerCxCxCx final : public BaseMQSSPass<SwapToLowerCxCxCx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwapToLowerCxCxCx)

  StringRef getArgument() const override { return "SwapToLowerCxCxCx"; }

  StringRef getDescription() const override {
    return "Decomposition pass of swap by three cx gates";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    auto ctx = kernel.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceSwapToLowerCxCxCx>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::SwapOp>();
    if (failed(applyPartialConversion(kernel, target, std::move(patterns)))) {
      kernel.emitOpError("SwapToLowerCxCxCxPass failed");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSwapToLowerCxCxCxPass() {
  return std::make_unique<SwapToLowerCxCxCx>();
}