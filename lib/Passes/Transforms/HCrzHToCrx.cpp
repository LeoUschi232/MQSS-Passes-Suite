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
#define GEN_PASS_DEF_HCRZHTOCRX

#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void foldHCrzHToCrx(Operation *op) {
  auto h2 = dyn_cast_or_null<quake::HOp>(*op);
  if (!h2 || !h2.getControls().empty() || h2.getTargets().size() != 1) {
    return;
  }
  auto prev =
      supportQuake::getPreviousOperationOnTarget(h2, h2.getTargets()[0]);
  if (!prev) {
    return;
  }
  auto crz = dyn_cast_or_null<quake::RzOp>(prev);
  if (!crz || crz.getControls().size() != 1 || crz.getTargets().size() != 1) {
    return;
  }
  auto idx2 = supportQuake::extractIndexFromQuakeExtractRefOp(
      h2.getTargets()[0].getDefiningOp());
  auto idx1 = supportQuake::extractIndexFromQuakeExtractRefOp(
      crz.getTargets()[0].getDefiningOp());
  if (idx1 != idx2) {
    return;
  }
  auto prev2 =
      supportQuake::getPreviousOperationOnTarget(crz, crz.getTargets()[0]);
  if (!prev2) {
    return;
  }
  auto h1 = dyn_cast_or_null<quake::HOp>(prev2);
  if (!h1 || !h1.getControls().empty() || h1.getTargets().size() != 1) {
    return;
  }
  mlir::IRRewriter rewriter(crz->getContext());
  rewriter.setInsertionPointAfter(crz);
  rewriter.create<quake::RxOp>(crz.getLoc(), crz.isAdj(), crz.getParameters(),
                               crz.getControls(), crz.getTargets());
  rewriter.eraseOp(h2);
  rewriter.eraseOp(crz);
  rewriter.eraseOp(h1);
}

class HCrzHToCrx : public BaseMQSSPass<HCrzHToCrx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCrzHToCrx)

  llvm::StringRef getArgument() const override { return "HCrzHToCrx"; }

  llvm::StringRef getDescription() const override {
    return "Fold H CRz H to CRx";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldHCrzHToCrx(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHCrzHToCrxPass() {
  return std::make_unique<HCrzHToCrx>();
}
