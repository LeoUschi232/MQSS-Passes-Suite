#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_TDGTTOID
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {
void cancelTdgTToId(mlir::Operation *op) {
  auto t = dyn_cast_or_null<quake::TOp>(*op);
  if (!t || t.isAdj() || !t.getControls().empty() ||
      t.getTargets().size() != 1) {
    return;
  }
  auto prevOp =
      supportQuake::getPreviousOperationOnTarget(t, t.getTargets()[0]);
  if (!prevOp) {
    return;
  }
  auto prevGate = dyn_cast_or_null<quake::TOp>(*prevOp);
  if (!prevGate || !prevGate.isAdj() || !prevGate.getControls().empty() ||
      prevGate.getTargets().size() != 1) {
    return;
  }
  auto loc = prevGate.getLoc();
  auto params = prevGate.getParameters();
  auto ctrls = prevGate.getControls();
  auto targs = prevGate.getTargets();

  mlir::IRRewriter rewriter(t->getContext());
  rewriter.setInsertionPointAfter(t);
  rewriter.eraseOp(t);
  rewriter.eraseOp(prevGate);
}

class TdgTToId : public BaseMQSSPass<TdgTToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TdgTToId)

  llvm::StringRef getArgument() const override { return "TdgTToId"; }

  llvm::StringRef getDescription() const override {
    return "Remove Tdg followed by T.";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) { cancelTdgTToId(op); });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createTdgTToIdPass() {
  return std::make_unique<TdgTToId>();
}
