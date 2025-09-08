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
#define GEN_PASS_DEF_CRXTOHCRZH

#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
struct ReplaceCrxToHCrzH : public OpRewritePattern<quake::RxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::RxOp crxOp,
                                PatternRewriter &rewriter) const override {
    if (crxOp.getControls().size() != 1 || crxOp.getTargets().size() != 1 ||
        crxOp.getParameters().size() != 1 || crxOp.isAdj()) {
      return success();
    }
    Value control = crxOp.getControls()[0];
    Value target = crxOp.getTargets()[0];
    auto param = crxOp.getParameters()[0];
    Location loc = crxOp.getLoc();
    rewriter.create<quake::HOp>(loc, target);
    rewriter.create<quake::RzOp>(loc, false, param, control, target);
    rewriter.create<quake::HOp>(loc, target);
    rewriter.replaceOp(crxOp, {});
    return success();
  }
};

class CrxToHCrzH : public BaseMQSSPass<CrxToHCrzH> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CrxToHCrzH)

  llvm::StringRef getArgument() const override { return "CrxToHCrzH"; }

  llvm::StringRef getDescription() const override {
    return "Decomposition pass of crx by h, crz and h";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    auto ctx = kernel.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceCrxToHCrzH>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::RxOp>();
    if (failed(applyPartialConversion(kernel, target, std::move(patterns)))) {
      kernel.emitOpError("CrxToHCrzHPass failed");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCrxToHCrzHPass() {
  return std::make_unique<CrxToHCrzH>();
}
