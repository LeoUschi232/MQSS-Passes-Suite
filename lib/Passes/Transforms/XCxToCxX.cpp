#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/CommutateOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_XCXTOCXX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class XCxToCxX final : public BaseMQSSPass<XCxToCxX> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XCxToCxX)

  StringRef getArgument() const override { return "XCxToCxX"; }

  StringRef getDescription() const override {
    return "Apply commutation pass to pattern X-Cx to Cx-X";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto xOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!xOp
          || xOp.getTargets().size() != 1
          || !xOp.getControls().empty()) {
        return;
      }
      auto optional_cxOp
          = getNextOperationOnTarget(xOp, xOp.getTargets()[0]);
      if (!optional_cxOp) {
        return;
      }
      auto cxOp = dyn_cast_or_null<quake::XOp>(*optional_cxOp);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1) {
        return;
      }
      IRRewriter rewriter(xOp->getContext());
      rewriter.setInsertionPointAfter(cxOp);
      ValueRange targets = xOp.getTargets();
      ValueRange controls = cxOp.getControls();
      Location loc = xOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.create<quake::XOp>(loc, false, targets);
      rewriter.eraseOp(xOp);
      rewriter.eraseOp(cxOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createXCxToCxXPass() {
  return std::make_unique<XCxToCxX>();
}