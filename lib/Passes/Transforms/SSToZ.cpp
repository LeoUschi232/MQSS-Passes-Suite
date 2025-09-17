#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_SSTOZ

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;

namespace {
void foldSSToZ(Operation *op) {
  auto s = dyn_cast_or_null<quake::SOp>(*op);
  {
    if (!s || s.getControls().size() != 0 || s.getTargets().size() != 1)
      return;
  }
  auto prev = supportQuake::getPreviousOperationOnTarget(s, s.getTargets()[0]);
  if (!prev) {
    return;
  }
  auto s2 = dyn_cast_or_null<quake::SOp>(prev);
  if (!s2 || s2.getControls().size() != 0 || s2.getTargets().size() != 1) {
    return;
  }
  IRRewriter rewriter(s->getContext());
  rewriter.setInsertionPointAfter(s);
  rewriter.create<quake::ZOp>(s.getLoc(), s.getTargets()[0]);
  rewriter.eraseOp(s);
  rewriter.eraseOp(s2);
}

class SSToZ final : public BaseMQSSPass<SSToZ> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SSToZ)

  StringRef getArgument() const override { return "SSToZ"; }

  StringRef getDescription() const override { return "Replace S S by Z"; }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) { foldSSToZ(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSSToZPass() {
  return std::make_unique<SSToZ>();
}
