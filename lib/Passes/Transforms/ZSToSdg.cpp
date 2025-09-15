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

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_ZSTOSDG

#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

void ReplaceZSToSdg(Operation *op) {
  auto s = dyn_cast_or_null<quake::SOp>(*op);
  if (!s || !s.getControls().empty() || s.getTargets().size() != 1 ||
      s.isAdj()) {
    return;
  }
  auto prevOp =
      supportQuake::getPreviousOperationOnTarget(s, s.getTargets()[0]);
  if (!prevOp) {
    return;
  }
  auto z = dyn_cast_or_null<quake::ZOp>(*prevOp);
  if (!z || !z.getControls().empty() || z.getTargets().size() != 1) {
    return;
  }
  IRRewriter rewriter(s->getContext());
  rewriter.setInsertionPointAfter(s);
  rewriter.create<quake::SOp>(s.getLoc(), true, s.getParameters(),
                              s.getControls(), s.getTargets());
  rewriter.eraseOp(s);
  rewriter.eraseOp(z);
}

class ZSToSdg final : public BaseMQSSPass<ZSToSdg> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZSToSdg)

  StringRef getArgument() const override { return "ZSToSdg"; }

  StringRef getDescription() const override {
    return "Optimization pass that replaces a pattern composed of S and Z by "
           "Sdg";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) { ReplaceZSToSdg(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZSToSdgPass() {
  return std::make_unique<ZSToSdg>();
}
