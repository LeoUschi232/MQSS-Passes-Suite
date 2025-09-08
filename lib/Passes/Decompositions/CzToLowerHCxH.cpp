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
  date   December 2024
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
#define GEN_PASS_DEF_CZTOLOWERHCXH

#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

struct ReplaceCzToLowerHCxH : public OpRewritePattern<quake::ZOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ZOp czOp,
                                PatternRewriter &rewriter) const override {
    if (czOp.getControls().size() != 1 || czOp.getTargets().size() != 1) {
      return success();
    }
    Value control = czOp.getControls()[0];
    Value target = czOp.getTargets()[0];
    Location loc = czOp.getLoc();
    rewriter.create<quake::HOp>(loc, control);
    rewriter.create<quake::XOp>(loc, target, control);
    rewriter.create<quake::HOp>(loc, control);
    rewriter.replaceOp(czOp, {});
    return success();
  }
};

class CzToLowerHCxH : public BaseMQSSPass<CzToLowerHCxH> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CzToLowerHCxH)

  llvm::StringRef getArgument() const override { return "CzToLowerHCxH"; }

  llvm::StringRef getDescription() const override {
    return "Decomposition pass of Cz by H, Cx, and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    auto ctx = kernel.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceCzToLowerHCxH>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::ZOp>();
    if (failed(applyPartialConversion(kernel, target, std::move(patterns)))) {
      kernel.emitOpError("CzToLowerHCxHPass failed");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCzToLowerHCxHPass() {
  return std::make_unique<CzToLowerHCxH>();
}
