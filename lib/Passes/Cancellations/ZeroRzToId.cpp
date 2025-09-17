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
#include "Passes/Cancellations.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_ZERORZTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {
void removeZeroRzToId(Operation *currentOp) {
  if (!isa<quake::RzOp>(currentOp)) {
    return;
  }
  auto gate = dyn_cast<quake::OperatorInterface>(currentOp);

  // Assume that parameters are all rotation angles
  bool deleteGate = true;
  for (auto parameter : gate.getParameters()) {
    if (double param = extractDoubleArgumentValue(parameter.getDefiningOp());
      !isMultipleOfTwoPi(param) && param != 0) {
      deleteGate = false;
    }
  }
  if (deleteGate) {
    IRRewriter rewriter(gate->getContext());
    rewriter.eraseOp(gate);
  }
}

class ZeroRzToId final : public BaseMQSSPass<ZeroRzToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroRzToId)

  StringRef getArgument() const override { return "ZeroRzToId"; }

  StringRef getDescription() const override {
    return "Optimization pass that removes Rz rotations with zero angles";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) { removeZeroRzToId(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZeroRzToIdPass() {
  return std::make_unique<ZeroRzToId>();
}