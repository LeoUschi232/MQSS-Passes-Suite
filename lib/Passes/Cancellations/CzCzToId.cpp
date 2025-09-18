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
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CZCZTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CzCzToId final : public BaseMQSSPass<CzCzToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CzCzToId)

  StringRef getArgument() const override { return "CzCzToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive identical CNOT gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto czOp1 = dyn_cast_or_null<quake::ZOp>(*op);
      if (!czOp1
          || czOp1.getTargets().size() != 1
          || czOp1.getControls().size() != 1) {
        return;
      }
      auto optional_czOp2
          = getNextOperationOnTarget(czOp1, czOp1.getTargets()[0]);
      if (!optional_czOp2) {
        return;
      }
      auto czOp2 = dyn_cast_or_null<quake::ZOp>(*optional_czOp2);
      if (!czOp2
          || czOp2.getTargets().size() != 1
          || czOp2.getControls().size() != 1
          || czOp2.getControls()[0] != czOp1.getControls()[0]) {
        return;
      }
      IRRewriter rewriter(czOp1->getContext());
      rewriter.eraseOp(czOp1);
      rewriter.eraseOp(czOp2);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCzCzToIdPass() {
  return std::make_unique<CzCzToId>();
}