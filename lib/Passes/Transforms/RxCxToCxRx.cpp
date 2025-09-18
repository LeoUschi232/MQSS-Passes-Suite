#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/CommutateOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_RXCXTOCXRX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class RxCxToCxRx final : public BaseMQSSPass<RxCxToCxRx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RxCxToCxRx)

  StringRef getArgument() const override { return "RxCxToCxRx"; }

  StringRef getDescription() const override {
    return "Apply commutation pass to pattern Rx-CNot to CNot-Rx";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto rxOp = dyn_cast_or_null<quake::RxOp>(op);
      if (!rxOp
          || rxOp.getTargets().size() != 1
          || !rxOp.getControls().empty()
          || rxOp.getParameters().size() != 1) {
        return;
      }
      auto optional_cxOp
          = getNextOperationOnTarget(rxOp, rxOp.getTargets()[0]);
      if (!optional_cxOp) {
        return;
      }
      auto cxOp = dyn_cast_or_null<quake::XOp>(optional_cxOp);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1) {
        return;
      }
      IRRewriter rewriter(rxOp->getContext());
      rewriter.setInsertionPointAfter(cxOp);
      ValueRange targets = cxOp.getTargets();
      ValueRange controls = cxOp.getControls();
      ValueRange parameters = rxOp.getParameters();
      Location loc = rxOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.create<quake::RxOp>(loc, parameters, ValueRange{}, targets);
      rewriter.eraseOp(rxOp);
      rewriter.eraseOp(cxOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createRxCxToCxRxPass() {
  return std::make_unique<RxCxToCxRx>();
}