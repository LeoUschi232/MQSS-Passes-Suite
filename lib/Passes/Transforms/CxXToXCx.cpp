#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/CommutateOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CXXTOXCX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CxXToXCx final : public BaseMQSSPass<CxXToXCx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxXToXCx)

  StringRef getArgument() const override { return "CxXToXCx"; }

  StringRef getDescription() const override {
    return "Apply commutation pass to pattern CNot-X to X-CNot";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto cxOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1) {
        return;
      }
      auto optional_xOp
          = getNextOperationOnTarget(cxOp, cxOp.getTargets()[0]);
      if (!optional_xOp) {
        return;
      }
      auto xOp = dyn_cast_or_null<quake::XOp>(*optional_xOp);
      if (!xOp
          || xOp.getTargets().size() != 1
          || !xOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(cxOp->getContext());
      rewriter.setInsertionPointAfter(xOp);
      ValueRange targets = cxOp.getTargets();
      ValueRange controls = cxOp.getControls();
      Location loc = cxOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, targets);
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.eraseOp(cxOp);
      rewriter.eraseOp(xOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxXToXCxPass() {
  return std::make_unique<CxXToXCx>();
}