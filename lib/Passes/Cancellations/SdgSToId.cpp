#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_SDGSTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class SdgSToId final : public BaseMQSSPass<SdgSToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdgSToId)

  StringRef getArgument() const override { return "SdgSToId"; }

  StringRef getDescription() const override {
    return "Remove Sdg followed by S.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto sOp1 = dyn_cast_or_null<quake::SOp>(*op);
      if (!sOp1
          || !sOp1.isAdj()
          || sOp1.getTargets().size() != 1
          || !sOp1.getControls().empty()) {
        return;
      }
      auto optional_sOp2 =
          getNextOperationOnTarget(sOp1, sOp1.getTargets()[0]);
      if (!optional_sOp2) {
        return;
      }
      auto sOp2 = dyn_cast_or_null<quake::SOp>(*optional_sOp2);
      if (!sOp2
          || sOp2.isAdj()
          || sOp2.getTargets().size() != 1
          || !sOp2.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(sOp1->getContext());
      rewriter.eraseOp(sOp1);
      rewriter.eraseOp(sOp2);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSdgSToIdPass() {
  return std::make_unique<SdgSToId>();
}