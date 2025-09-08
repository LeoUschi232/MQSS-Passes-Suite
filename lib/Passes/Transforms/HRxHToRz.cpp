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
#define GEN_PASS_DEF_HRXHTORZ

#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void foldHRxH(Operation *op) {
  auto h2 = dyn_cast_or_null<quake::HOp>(*op);
  if (!h2 || h2.getControls().size() != 0 || h2.getTargets().size() != 1)
    return;
  auto prev =
      supportQuake::getPreviousOperationOnTarget(h2, h2.getTargets()[0]);
  if (!prev)
    return;
  auto rx = dyn_cast_or_null<quake::RxOp>(prev);
  if (!rx || rx.getControls().size() != 0 || rx.getTargets().size() != 1)
    return;
  auto prev2 =
      supportQuake::getPreviousOperationOnTarget(rx, h2.getTargets()[0]);
  if (!prev2)
    return;
  auto h1 = dyn_cast_or_null<quake::HOp>(prev2);
  if (!h1 || h1.getControls().size() != 0 || h1.getTargets().size() != 1)
    return;
  mlir::IRRewriter rewriter(h2->getContext());
  rewriter.setInsertionPointAfter(h2);
  rewriter.create<quake::RzOp>(h2.getLoc(), rx.isAdj(), rx.getParameters(),
                               rx.getControls(), rx.getTargets());
  rewriter.eraseOp(h2);
  rewriter.eraseOp(rx);
  rewriter.eraseOp(h1);
}

class HRxHToRz : public BaseMQSSPass<HRxHToRz> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HRxHToRz)

  llvm::StringRef getArgument() const override { return "HRxHToRz"; }

  llvm::StringRef getDescription() const override {
    return "Fold H Rx H to Rz";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldHRxH(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHRxHToRzPass() {
  return std::make_unique<HRxHToRz>();
}
