#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_YHTOHY

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class YHToHY final : public BaseMQSSPass<YHToHY> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(YHToHY)
  StringRef getArgument() const override { return "YHToHY"; }

  StringRef getDescription() const override {
    return "Switches a pattern composed by Y H to H Y";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto yOp = dyn_cast_or_null<quake::YOp>(*op);
      if (!yOp
          || yOp.getTargets().size() != 1
          || !yOp.getControls().empty()) {
        return;
      }
      auto optional_hOp
          = getNextOperationOnTarget(yOp, yOp.getTargets()[0]);
      if (!optional_hOp) {
        return;
      }
      auto hOp = dyn_cast_or_null<quake::HOp>(*optional_hOp);
      if (!hOp
          || hOp.getTargets().size() != 1
          || !hOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(yOp->getContext());
      rewriter.setInsertionPointAfter(hOp);
      Value target = yOp.getTargets()[0];
      Location loc = yOp.getLoc();
      rewriter.create<quake::HOp>(loc, false, target);
      rewriter.create<quake::YOp>(loc, false, target);
      rewriter.eraseOp(yOp);
      rewriter.eraseOp(hOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createYHToHYPass() {
  return std::make_unique<YHToHY>();
}