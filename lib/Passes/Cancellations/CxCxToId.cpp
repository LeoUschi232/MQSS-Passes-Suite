#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CXCXTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CxCxToId final : public BaseMQSSPass<CxCxToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxCxToId)

  StringRef getArgument() const override { return "CxCxToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive identical Cx gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto cxOp1 = dyn_cast_or_null<quake::XOp>(op);
      if (!cxOp1
          || cxOp1.getTargets().size() != 1
          || cxOp1.getControls().size() != 1) {
        return;
      }
      auto optional_cxOp2_onTarget
          = getNextOperationOnTarget(cxOp1, cxOp1.getTargets()[0]);
      auto optional_cxOp2_onControl
          = getNextOperationOnTarget(cxOp1, cxOp1.getControls()[0]);
      if (!optional_cxOp2_onTarget
          || !optional_cxOp2_onControl
          || optional_cxOp2_onTarget != optional_cxOp2_onControl) {
        return;
      }
      auto cxOp2
          = dyn_cast_or_null<quake::XOp>(optional_cxOp2_onTarget);
      if (!cxOp2
          || cxOp2.getTargets().size() != 1
          || cxOp2.getControls().size() != 1
          || cxOp2.getControls()[0] != cxOp1.getControls()[0]) {
        return;
      }
      IRRewriter rewriter(cxOp1->getContext());
      rewriter.eraseOp(cxOp1);
      rewriter.eraseOp(cxOp2);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxCxToIdPass() {
  return std::make_unique<CxCxToId>();
}