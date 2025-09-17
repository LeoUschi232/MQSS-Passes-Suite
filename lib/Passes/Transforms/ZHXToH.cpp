/* Fold pattern Z H X -> H */
#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_ZHXTOH

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {
void ReplaceZHXToH(Operation *op) {
  auto x = dyn_cast_or_null<quake::XOp>(*op);
  if (!x || !x.getControls().empty() || x.getTargets().size() != 1) {
    return;
  }
  auto prevHOp =
      supportQuake::getPreviousOperationOnTarget(x, x.getTargets()[0]);
  if (!prevHOp) {
    return;
  }
  auto h = dyn_cast_or_null<quake::HOp>(prevHOp);
  if (!h || !h.getControls().empty() || h.getTargets().size() != 1) {
    return;
  }
  auto prevZOp =
      supportQuake::getPreviousOperationOnTarget(h, h.getTargets()[0]);
  if (!prevZOp) {
    return;
  }
  auto z = dyn_cast_or_null<quake::ZOp>(prevZOp);
  if (!z || !z.getControls().empty() || z.getTargets().size() != 1) {
    return;
  }
  IRRewriter rewriter(x->getContext());
  rewriter.setInsertionPoint(x);
  rewriter.create<quake::HOp>(x.getLoc(), x.getTargets());
  rewriter.eraseOp(x);
  rewriter.eraseOp(h);
  rewriter.eraseOp(z);
}

class ZHXToH final : public BaseMQSSPass<ZHXToH> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHXToH)

  StringRef getArgument() const override { return "ZHXToH"; }

  StringRef getDescription() const override { return "Fold Z H X to H"; }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) { ReplaceZHXToH(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZHXToHPass() {
  return std::make_unique<ZHXToH>();
}