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
#define GEN_PASS_DEF_SDGSDGTOZ

#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void foldSdgSdgToZ(Operation *op) {
  auto sdg = dyn_cast_or_null<quake::SOp>(*op);
  if (!sdg || !sdg.isAdj() || sdg.getControls().size() != 0 ||
      sdg.getTargets().size() != 1)
    return;
  auto prev =
      supportQuake::getPreviousOperationOnTarget(sdg, sdg.getTargets()[0]);
  if (!prev)
    return;
  auto sdg2 = dyn_cast_or_null<quake::SOp>(prev);
  if (!sdg2 || !sdg2.isAdj() || sdg2.getControls().size() != 0 ||
      sdg2.getTargets().size() != 1)
    return;
  mlir::IRRewriter rewriter(sdg->getContext());
  rewriter.setInsertionPointAfter(sdg);
  rewriter.create<quake::ZOp>(sdg.getLoc(), sdg.getTargets()[0]);
  rewriter.eraseOp(sdg);
  rewriter.eraseOp(sdg2);
}

class SdgSdgToZ : public BaseMQSSPass<SdgSdgToZ> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdgSdgToZ)

  llvm::StringRef getArgument() const override { return "SdgSdgToZ"; }

  llvm::StringRef getDescription() const override {
    return "Replace Sdg Sdg by Z";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldSdgSdgToZ(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSdgSdgToZPass() {
  return std::make_unique<SdgSdgToZ>();
}
