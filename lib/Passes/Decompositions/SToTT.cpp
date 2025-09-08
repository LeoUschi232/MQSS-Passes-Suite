/* This code and any associated documentation is provided "as is"
 *
 * Copyright 2024 Munich Quantum Software Stack Project
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *****************************************************************************
   author Martin Letras
   date   February 2025
   version 1.0
 ***************************************************************************/

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
#define GEN_PASS_DEF_STOTT

#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

struct ReplaceSToTT : public OpRewritePattern<quake::SOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::SOp sOp,
                                PatternRewriter &rewriter) const override {
    if (sOp.isAdj() || sOp.getControls().size() != 0 ||
        sOp.getTargets().size() != 1) {
      return success();
    }
    auto loc = sOp.getLoc();
    auto target = sOp.getTargets()[0];
    rewriter.create<quake::TOp>(loc, false, target);
    rewriter.create<quake::TOp>(loc, false, target);
    rewriter.replaceOp(sOp, {});
    return success();
  }
};

class SToTT : public BaseMQSSPass<SToTT> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SToTT)

  llvm::StringRef getArgument() const override { return "SToTT"; }

  llvm::StringRef getDescription() const override {
    return "Decomposition pass that replaces S by two T gates";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    auto ctx = kernel.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceSToTT>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::SOp>();
    if (failed(applyPartialConversion(kernel, target, std::move(patterns)))) {
      kernel.emitOpError("SToTT decomposition failed");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createSToTTPass() {
  return std::make_unique<SToTT>();
}
