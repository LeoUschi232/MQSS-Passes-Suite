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
 ******************************************************************************
author Martin Letras date February 2025 version 1.0 *
*****************************************************************************/

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
#define GEN_PASS_DEF_ZTOHXH
#include "Passes/Decompositions.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {

struct ReplaceZToHXH : public OpRewritePattern<quake::ZOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ZOp zOp,
                                PatternRewriter &rewriter) const override {
    if (zOp.isAdj() || !zOp.getControls().empty() ||
        zOp.getTargets().size() != 1) {
      return success();
    }
    auto loc = zOp.getLoc();
    auto target = zOp.getTargets()[0];
    rewriter.create<quake::HOp>(loc, target);
    rewriter.create<quake::XOp>(loc, false, target);
    rewriter.create<quake::HOp>(loc, target);
    rewriter.replaceOp(zOp, {});
    return success();
  }
};

class ZToHXH : public BaseMQSSPass<ZToHXH> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZToHXH)

  llvm::StringRef getArgument() const override { return "ZToHXH"; }

  llvm::StringRef getDescription() const override {
    return "Decomposition pass of Z by H, X and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    auto ctx = kernel.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceZToHXH>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::ZOp>();
    if (failed(applyPartialConversion(kernel, target, std::move(patterns)))) {
      kernel.emitOpError("ZToHXH decomposition failed");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createZToHXHPass() {
  return std::make_unique<ZToHXH>();
}
