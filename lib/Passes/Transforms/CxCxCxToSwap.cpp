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
#define GEN_PASS_DEF_CXCXCXTOSWAP

#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void foldCxCxCx(Operation *op) {
  auto cx3 = dyn_cast_or_null<quake::XOp>(*op);
  if (!cx3 || cx3.getControls().size() != 1 || cx3.getTargets().size() != 1)
    return;
  auto prev2 =
      supportQuake::getPreviousOperationOnTarget(cx3, cx3.getTargets()[0]);
  if (!prev2)
    return;
  auto cx2 = dyn_cast_or_null<quake::XOp>(prev2);
  if (!cx2 || cx2.getControls().size() != 1 || cx2.getTargets().size() != 1)
    return;
  auto prev1 =
      supportQuake::getPreviousOperationOnTarget(cx2, cx2.getTargets()[0]);
  if (!prev1)
    return;
  auto cx1 = dyn_cast_or_null<quake::XOp>(prev1);
  if (!cx1 || cx1.getControls().size() != 1 || cx1.getTargets().size() != 1)
    return;
  // check pattern control-target alternating
  int c1 = supportQuake::extractIndexFromQuakeExtractRefOp(
      cx1.getControls()[0].getDefiningOp());
  int t1 = supportQuake::extractIndexFromQuakeExtractRefOp(
      cx1.getTargets()[0].getDefiningOp());
  int c2 = supportQuake::extractIndexFromQuakeExtractRefOp(
      cx2.getControls()[0].getDefiningOp());
  int t2 = supportQuake::extractIndexFromQuakeExtractRefOp(
      cx2.getTargets()[0].getDefiningOp());
  int c3 = supportQuake::extractIndexFromQuakeExtractRefOp(
      cx3.getControls()[0].getDefiningOp());
  int t3 = supportQuake::extractIndexFromQuakeExtractRefOp(
      cx3.getTargets()[0].getDefiningOp());
  if (!(c1 == c3 && t1 == t3 && c2 == t1 && t2 == c1))
    return;
  mlir::IRRewriter rewriter(cx3->getContext());
  rewriter.setInsertionPointAfter(cx3);
  SmallVector<Value> tgts{cx3.getControls()[0], cx3.getTargets()[0]};
  rewriter.create<quake::SwapOp>(cx3.getLoc(), /*params*/ ValueRange{},
                                 ValueRange{}, tgts);
  rewriter.eraseOp(cx3);
  rewriter.eraseOp(cx2);
  rewriter.eraseOp(cx1);
}

class CxCxCxToSwap : public BaseMQSSPass<CxCxCxToSwap> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxCxCxToSwap)

  llvm::StringRef getArgument() const override { return "CxCxCxToSwap"; }

  llvm::StringRef getDescription() const override {
    return "Replace CNOT CNOT CNOT swap pattern by SWAP";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldCxCxCx(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxCxCxToSwapPass() {
  return std::make_unique<CxCxCxToSwap>();
}
