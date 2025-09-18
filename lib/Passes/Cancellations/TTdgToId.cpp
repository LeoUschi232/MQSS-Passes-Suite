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
#define GEN_PASS_DEF_TTDGTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class TTdgToId final : public BaseMQSSPass<TTdgToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTdgToId)

  StringRef getArgument() const override { return "TTdgToId"; }

  StringRef getDescription() const override {
    return "Remove Tdg followed by T.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto tOp1 = dyn_cast_or_null<quake::TOp>(*op);
      if (!tOp1
          || tOp1.isAdj()
          || tOp1.getTargets().size() != 1
          || !tOp1.getControls().empty()) {
        return;
      }
      auto optional_tOp2
          = getNextOperationOnTarget(tOp1, tOp1.getTargets()[0]);
      if (!optional_tOp2) {
        return;
      }
      auto tOp2 = dyn_cast_or_null<quake::TOp>(*optional_tOp2);
      if (!tOp2
          || !tOp2.isAdj()
          || tOp2.getTargets().size() != 1
          || !tOp2.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(tOp1->getContext());
      rewriter.eraseOp(tOp1);
      rewriter.eraseOp(tOp2);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createTTdgToIdPass() {
  return std::make_unique<TTdgToId>();
}