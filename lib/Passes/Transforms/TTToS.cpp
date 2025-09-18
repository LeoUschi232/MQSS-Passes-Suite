#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_TTTOS

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class TTToS final : public BaseMQSSPass<TTToS> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTToS)

  StringRef getArgument() const override { return "TTToS"; }

  StringRef getDescription() const override { return "Replace T T by S"; }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto tOp1 = dyn_cast_or_null<quake::TOp>(op);
      if (!tOp1
          || tOp1.isAdj()
          || tOp1.getTargets().size() != 1
          || !tOp1.getControls().empty()) {
        return;
      }
      auto optional_tOp
          = getNextOperationOnTarget(tOp1, tOp1.getTargets()[0]);
      if (!optional_tOp) {
        return;
      }
      auto tOp2 = dyn_cast_or_null<quake::TOp>(optional_tOp);
      if (!tOp2
          || tOp2.isAdj()
          || tOp2.getTargets().size() != 1
          || !tOp2.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(tOp1->getContext());
      rewriter.setInsertionPointAfter(tOp2);
      Location loc = tOp1.getLoc();
      ValueRange targets = tOp1.getTargets();
      rewriter.create<quake::SOp>(loc, false, targets);
      rewriter.eraseOp(tOp1);
      rewriter.eraseOp(tOp2);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createTTToSPass() {
  return std::make_unique<TTToS>();
}