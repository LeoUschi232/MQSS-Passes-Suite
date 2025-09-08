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
#define GEN_PASS_DEF_HCRXHTOCRZ

#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void foldHCrxHToCrz(Operation *op) {
  auto h2 = dyn_cast_or_null<quake::HOp>(*op);
  if (!h2 || !h2.getControls().empty() || h2.getTargets().size() != 1) {
    return;
  }
  auto prev =
      supportQuake::getPreviousOperationOnTarget(h2, h2.getTargets()[0]);
  if (!prev) {
    return;
  }
  auto crx = dyn_cast_or_null<quake::RxOp>(prev);
  if (!crx || crx.getControls().size() != 1 || crx.getTargets().size() != 1) {
    return;
  }
  auto idx2 = supportQuake::extractIndexFromQuakeExtractRefOp(
      h2.getTargets()[0].getDefiningOp());
  auto idx1 = supportQuake::extractIndexFromQuakeExtractRefOp(
      crx.getTargets()[0].getDefiningOp());
  if (idx1 != idx2) {
    return;
  }
  auto prev2 =
      supportQuake::getPreviousOperationOnTarget(crx, crx.getTargets()[0]);
  if (!prev2) {
    return;
  }
  auto h1 = dyn_cast_or_null<quake::HOp>(prev2);
  if (!h1 || !h1.getControls().empty() || h1.getTargets().size() != 1) {
    return;
  }
  mlir::IRRewriter rewriter(crx->getContext());
  rewriter.setInsertionPointAfter(crx);
  rewriter.create<quake::RzOp>(crx.getLoc(), crx.isAdj(), crx.getParameters(),
                               crx.getControls(), crx.getTargets());
  rewriter.eraseOp(h2);
  rewriter.eraseOp(crx);
  rewriter.eraseOp(h1);
}

class HCrxHToCrz : public BaseMQSSPass<HCrxHToCrz> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCrxHToCrz)

  llvm::StringRef getArgument() const override { return "HCrxHToCrz"; }

  llvm::StringRef getDescription() const override {
    return "Fold H CRx H to CRz";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldHCrxHToCrz(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHCrxHToCrzPass() {
  return std::make_unique<HCrxHToCrz>();
}
