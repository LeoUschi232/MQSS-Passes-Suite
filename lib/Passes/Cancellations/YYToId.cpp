#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_YYTOID
// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class YYToId final : public BaseMQSSPass<YYToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(YYToId)

  StringRef getArgument() const override { return "YYToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive Y gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto yOp1 = dyn_cast_or_null<quake::YOp>(op);
      if (!yOp1
          || yOp1.isAdj()
          || yOp1.getTargets().size() != 1
          || !yOp1.getControls().empty()) {
        return;
      }
      auto optional_yOp2
          = getNextOperationOnTarget(yOp1, yOp1.getTargets()[0]);
      if (!optional_yOp2) {
        return;
      }
      auto yOp2 = dyn_cast_or_null<quake::YOp>(optional_yOp2);
      if (!yOp2
          || yOp2.isAdj()
          || yOp2.getTargets().size() != 1
          || !yOp2.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(yOp1->getContext());
      rewriter.eraseOp(yOp1);
      rewriter.eraseOp(yOp2);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createYYToIdPass() {
  return std::make_unique<YYToId>();
}