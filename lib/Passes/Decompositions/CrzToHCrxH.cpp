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
#define GEN_PASS_DEF_CRZTOHCRXH

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
struct ReplaceCrzToHCrxH final : OpRewritePattern<quake::RzOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::RzOp crzOp,
                                PatternRewriter &rewriter) const override {
    if (crzOp.getControls().size() != 1 || crzOp.getTargets().size() != 1 ||
        crzOp.getParameters().size() != 1 || crzOp.isAdj()) {
      return success();
    }
    Value control = crzOp.getControls()[0];
    Value target = crzOp.getTargets()[0];
    Location loc = crzOp.getLoc();
    auto param = crzOp.getParameters()[0];
    rewriter.create<quake::HOp>(loc, target);
    rewriter.create<quake::RxOp>(loc, false, param, control, target);
    rewriter.create<quake::HOp>(loc, target);
    rewriter.replaceOp(crzOp, {});
    return success();
  }
};

class CrzToHCrxH final : public BaseMQSSPass<CrzToHCrxH> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CrzToHCrxH)

  StringRef getArgument() const override { return "CrzToHCrxH"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Crz by H, Crx, and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    auto ctx = kernel.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceCrzToHCrxH>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::RzOp>();
    if (failed(applyPartialConversion(kernel, target, std::move(patterns)))) {
      kernel.emitOpError("CrzToHCrxHPass failed");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCrzToHCrxHPass() {
  return std::make_unique<CrzToHCrxH>();
}