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
***************************************************************************
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
#define GEN_PASS_DEF_STOSDGSDGSDG

#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

void replaceSToSdgSdgSdg(Operation *currentOp) {
  auto sGate = dyn_cast_or_null<quake::SOp>(currentOp);
  if (!sGate || sGate.isAdj() || sGate.getControls().size() != 0 ||
      sGate.getTargets().size() != 1) {
    return;
  }
  auto loc = sGate.getLoc();
  auto target = sGate.getTargets()[0];
  mlir::IRRewriter rewriter(sGate->getContext());
  rewriter.setInsertionPointAfter(sGate);
  rewriter.create<quake::SOp>(loc, true, target);
  rewriter.create<quake::SOp>(loc, true, target);
  rewriter.create<quake::SOp>(loc, true, target);
  rewriter.eraseOp(sGate);
}

class SToSdgSdgSdg : public BaseMQSSPass<SToSdgSdgSdg> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SToSdgSdgSdg)

  llvm::StringRef getArgument() const override { return "SToSdgSdgSdg"; }

  llvm::StringRef getDescription() const override {
    return "Decomposition pass that replaces an S gate by Sdg, Sdg and Sdg";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { replaceSToSdgSdgSdg(op); });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createSToSdgSdgSdgPass() {
  return std::make_unique<SToSdgSdgSdg>();
}
