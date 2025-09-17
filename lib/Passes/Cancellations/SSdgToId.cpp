#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_SSDGTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {
void cancelSSdgToId(Operation *op) {
  auto s = dyn_cast_or_null<quake::SOp>(*op);
  if (!s || !s.isAdj() || !s.getControls().empty() ||
      s.getTargets().size() != 1) {
    return;
  }
  auto prevOp =
      supportQuake::getPreviousOperationOnTarget(s, s.getTargets()[0]);
  if (!prevOp) {
    return;
  }
  auto prevGate = dyn_cast_or_null<quake::SOp>(*prevOp);
  if (!prevGate || prevGate.isAdj() || !prevGate.getControls().empty() ||
      prevGate.getTargets().size() != 1) {
    return;
  }

  IRRewriter rewriter(s->getContext());
  rewriter.setInsertionPointAfter(s);
  rewriter.eraseOp(s);
  rewriter.eraseOp(prevGate);
}

class SSdgToId final : public BaseMQSSPass<SSdgToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SSdgToId)

  StringRef getArgument() const override { return "SSdgToId"; }

  StringRef getDescription() const override {
    return "Remove Sdg followed by S.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) { cancelSSdgToId(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSSdgToIdPass() {
  return std::make_unique<SSdgToId>();
}