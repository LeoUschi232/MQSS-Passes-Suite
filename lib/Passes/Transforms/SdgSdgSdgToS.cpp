#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_SDGSDGSDGTOS

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void foldSdgSdgSdgToS(Operation *op) {
  auto s = dyn_cast_or_null<quake::SOp>(*op);
  if (!s || !s.isAdj() || s.getControls().size() != 0 ||
      s.getTargets().size() != 1) {
    return;
  }
  auto prev1 = supportQuake::getPreviousOperationOnTarget(s, s.getTargets()[0]);
  if (!prev1) {
    return;
  }
  auto s2 = dyn_cast_or_null<quake::SOp>(prev1);
  if (!s2 || !s2.isAdj() || s2.getControls().size() != 0 ||
      s2.getTargets().size() != 1) {
    return;
  }
  auto prev2 =
      supportQuake::getPreviousOperationOnTarget(s2, s.getTargets()[0]);
  if (!prev2) {
    return;
  }
  auto s3 = dyn_cast_or_null<quake::SOp>(prev2);
  if (!s3 || !s3.isAdj() || s3.getControls().size() != 0 ||
      s3.getTargets().size() != 1) {
    return;
  }
  IRRewriter rewriter(s->getContext());
  rewriter.setInsertionPointAfter(s);
  rewriter.create<quake::SOp>(s.getLoc(), s.getTargets()[0]);
  rewriter.eraseOp(s);
  rewriter.eraseOp(s2);
  rewriter.eraseOp(s3);
}

class SdgSdgSdgToS final : public BaseMQSSPass<SdgSdgSdgToS> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdgSdgSdgToS)

  StringRef getArgument() const override { return "SdgSdgSdgToS"; }

  StringRef getDescription() const override {
    return "Replace Sdg Sdg Sdg by S";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldSdgSdgSdgToS(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSdgSdgSdgToSPass() {
  return std::make_unique<SdgSdgSdgToS>();
}