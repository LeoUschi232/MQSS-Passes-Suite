#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HCZHTOCX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void foldHCzHToCx(Operation *op) {
  auto h2 = dyn_cast_or_null<quake::HOp>(*op);
  if (!h2 || !h2.getControls().empty() || h2.getTargets().size() != 1) {
    return;
  }
  auto prev =
      supportQuake::getPreviousOperationOnTarget(h2, h2.getTargets()[0]);
  if (!prev) {
    return;
  }
  auto cz = dyn_cast_or_null<quake::ZOp>(prev);
  if (!cz || cz.getControls().size() != 1 || cz.getTargets().size() != 1) {
    return;
  }
  auto idx2 = supportQuake::extractIndexFromQuakeExtractRefOp(
      h2.getTargets()[0].getDefiningOp());
  auto idx1 = supportQuake::extractIndexFromQuakeExtractRefOp(
      cz.getTargets()[0].getDefiningOp());
  if (idx1 != idx2) {
    return;
  }
  auto prev2 =
      supportQuake::getPreviousOperationOnTarget(cz, cz.getTargets()[0]);
  if (!prev2) {
    return;
  }
  auto h1 = dyn_cast_or_null<quake::HOp>(prev2);
  if (!h1 || !h1.getControls().empty() || h1.getTargets().size() != 1) {
    return;
  }
  IRRewriter rewriter(cz->getContext());
  rewriter.setInsertionPointAfter(cz);
  rewriter.create<quake::XOp>(cz.getLoc(), cz.getControls(), cz.getTargets());
  rewriter.eraseOp(h2);
  rewriter.eraseOp(cz);
  rewriter.eraseOp(h1);
}

class HCzHToCx final : public BaseMQSSPass<HCzHToCx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCzHToCx)

  StringRef getArgument() const override { return "HCzHToCx"; }

  StringRef getDescription() const override {
    return "Fold H Cz H to Cx";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldHCzHToCx(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHCzHToCxPass() {
  return std::make_unique<HCzHToCx>();
}