#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_TTTOS

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {
void foldTT(Operation *op) {
  auto t = dyn_cast_or_null<quake::TOp>(*op);
  if (!t || t.getControls().size() != 0 || t.getTargets().size() != 1) {
    return;
  }
  auto prev = supportQuake::getPreviousOperationOnTarget(t, t.getTargets()[0]);
  if (!prev) {
    return;
  }
  auto t2 = dyn_cast_or_null<quake::TOp>(prev);
  if (!t2 || t2.getControls().size() != 0 || t2.getTargets().size() != 1) {
    return;
  }
  IRRewriter rewriter(t->getContext());
  rewriter.setInsertionPointAfter(t);
  rewriter.create<quake::SOp>(t.getLoc(), t.getTargets()[0]);
  rewriter.eraseOp(t);
  rewriter.eraseOp(t2);
}

class TTToS final : public BaseMQSSPass<TTToS> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTToS)

  StringRef getArgument() const override { return "TTToS"; }

  StringRef getDescription() const override { return "Replace T T by S"; }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldTT(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createTTToSPass() {
  return std::make_unique<TTToS>();
}