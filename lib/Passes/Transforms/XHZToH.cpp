/* Fold pattern X H Z -> H */
#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_XHZTOH

#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void ReplaceXHZToH(Operation *op) {
  auto z = dyn_cast_or_null<quake::ZOp>(*op);
  if (!z || !z.getControls().empty() || z.getTargets().size() != 1) {
    return;
  }
  auto prevHOp =
      supportQuake::getPreviousOperationOnTarget(z, z.getTargets()[0]);
  if (!prevHOp) {
    return;
  }
  auto h = dyn_cast_or_null<quake::HOp>(prevHOp);
  if (!h || !h.getControls().empty() || h.getTargets().size() != 1) {
    return;
  }
  auto prevXOp =
      supportQuake::getPreviousOperationOnTarget(h, h.getTargets()[0]);
  if (!prevXOp) {
    return;
  }
  auto x = dyn_cast_or_null<quake::XOp>(prevXOp);
  if (!x || !x.getControls().empty() || x.getTargets().size() != 1) {
    return;
  }
  mlir::IRRewriter rewriter(z->getContext());
  rewriter.setInsertionPoint(z);
  rewriter.create<quake::HOp>(z.getLoc(), z.getTargets());
  rewriter.eraseOp(z);
  rewriter.eraseOp(h);
  rewriter.eraseOp(x);
}

class XHZToH : public BaseMQSSPass<XHZToH> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XHZToH)

  llvm::StringRef getArgument() const override { return "XHZToH"; }

  llvm::StringRef getDescription() const override {
    return "Optimization pass that replaces a pattern X H Z by H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { ReplaceXHZToH(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createXHZToHPass() {
  return std::make_unique<XHZToH>();
}