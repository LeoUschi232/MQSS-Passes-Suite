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
#define GEN_PASS_DEF_ZCXTOCXZ

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;


namespace {
class ZCxToCxZ final : public BaseMQSSPass<ZCxToCxZ> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZCxToCxZ)

  StringRef getArgument() const override { return "ZCxToCxZ"; }

  StringRef getDescription() const override {
    return "Apply commutation pass to pattern Z-CNot to CNot-Z";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto zOp = dyn_cast_or_null<quake::ZOp>(op);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      auto optional_cxOp
          = getNextOperationOnTarget(zOp, zOp.getTargets()[0]);
      if (!optional_cxOp) {
        return;
      }
      auto cxOp = dyn_cast_or_null<quake::XOp>(optional_cxOp);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1
          || cxOp.getControls()[0] != zOp.getTargets()[0]) {
        return;
      }
      IRRewriter rewriter(zOp->getContext());
      rewriter.setInsertionPointAfter(cxOp);
      ValueRange targets = cxOp.getTargets();
      ValueRange controls = cxOp.getControls();
      Location loc = cxOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.create<quake::ZOp>(loc, false, targets);
      rewriter.eraseOp(zOp);
      rewriter.eraseOp(cxOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZCxToCxZPass() {
  return std::make_unique<ZCxToCxZ>();
}