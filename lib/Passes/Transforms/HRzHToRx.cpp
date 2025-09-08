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
#define GEN_PASS_DEF_HRZHTORX

#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void foldHRzH(Operation *op) {
  auto h2 = dyn_cast_or_null<quake::HOp>(*op);
  if (!h2 || h2.getControls().size() != 0 || h2.getTargets().size() != 1)
    return;
  auto prev =
      supportQuake::getPreviousOperationOnTarget(h2, h2.getTargets()[0]);
  if (!prev)
    return;
  auto rz = dyn_cast_or_null<quake::RzOp>(prev);
  if (!rz || rz.getControls().size() != 0 || rz.getTargets().size() != 1)
    return;
  auto prev2 =
      supportQuake::getPreviousOperationOnTarget(rz, rz.getTargets()[0]);
  if (!prev2)
    return;
  auto h1 = dyn_cast_or_null<quake::HOp>(prev2);
  if (!h1 || h1.getControls().size() != 0 || h1.getTargets().size() != 1)
    return;
  mlir::IRRewriter rewriter(rz->getContext());
  rewriter.setInsertionPointAfter(rz);
  rewriter.create<quake::RxOp>(rz.getLoc(), rz.isAdj(), rz.getParameters(),
                               rz.getControls(), rz.getTargets());
  rewriter.eraseOp(h2);
  rewriter.eraseOp(rz);
  rewriter.eraseOp(h1);
}

class HRzHToRx : public BaseMQSSPass<HRzHToRx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HRzHToRx)

  llvm::StringRef getArgument() const override { return "HRzHToRx"; }

  llvm::StringRef getDescription() const override {
    return "Fold H Rz H to Rx";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldHRzH(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHRzHToRxPass() {
  return std::make_unique<HRzHToRx>();
}
