#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_HZHTOX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class HZHToX final : public BaseMQSSPass<HZHToX> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HZHToX)

  StringRef getArgument() const override { return "HZHToX"; }

  StringRef getDescription() const override {
    return "Optimization pass that replaces a pattern composed of H, Z, H by X";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto hOp1 = dyn_cast_or_null<quake::HOp>(op);
      if (!hOp1
          || hOp1.getTargets().size() != 1
          || !hOp1.getControls().empty()) {
        return;
      }
      auto optional_zOp
          = getNextOperationOnTarget(hOp1, hOp1.getTargets()[0]);
      if (!optional_zOp) {
        return;
      }
      auto zOp = dyn_cast_or_null<quake::ZOp>(optional_zOp);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      auto optional_hOp2
          = getNextOperationOnTarget(zOp, zOp.getTargets()[0]);
      if (!optional_hOp2) {
        return;
      }
      auto hOp2 = dyn_cast_or_null<quake::HOp>(optional_hOp2);
      if (!hOp2
          || hOp2.getTargets().size() != 1
          || !hOp2.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(hOp1->getContext());
      rewriter.setInsertionPointAfter(hOp2);
      ValueRange targets = hOp1.getTargets();
      Location loc = hOp1.getLoc();
      rewriter.create<quake::XOp>(loc, false, targets);
      rewriter.eraseOp(hOp1);
      rewriter.eraseOp(zOp);
      rewriter.eraseOp(hOp2);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHZHToXPass() {
  return std::make_unique<HZHToX>();
}