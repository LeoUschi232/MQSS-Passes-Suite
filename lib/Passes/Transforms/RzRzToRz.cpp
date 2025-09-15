#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_RZRZTORZ
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {

void foldRzRz(Operation *op, OpBuilder &builder) {
  auto rz2 = dyn_cast_or_null<quake::RzOp>(*op);
  if (!rz2 || !rz2.getControls().empty() || rz2.getTargets().size() != 1)
    return;
  auto prev =
      supportQuake::getPreviousOperationOnTarget(rz2, rz2.getTargets()[0]);
  if (!prev)
    return;
  auto rz1 = dyn_cast_or_null<quake::RzOp>(prev);
  if (!rz1 || !rz1.getControls().empty() || rz1.getTargets().size() != 1)
    return;
  builder.setInsertionPoint(rz2);
  auto p1 = supportQuake::getParametersValues(rz1.getParameters());
  auto p2 = supportQuake::getParametersValues(rz2.getParameters());
  if (p1.size() != p2.size())
    return;
  SmallVector<Value> params;
  for (size_t i = 0; i < p1.size(); ++i) {
    params.push_back(
        supportQuake::createFloatValue(builder, rz2.getLoc(), p1[i] + p2[i]));
  }
  IRRewriter rewriter(rz2->getContext());
  rewriter.setInsertionPointAfter(rz2);
  rewriter.create<quake::RzOp>(rz2.getLoc(), rz2.isAdj(), params,
                               rz2.getControls(), rz2.getTargets());
  rewriter.eraseOp(rz2);
  rewriter.eraseOp(rz1);
}

class RzRzToRz final : public BaseMQSSPass<RzRzToRz> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RzRzToRz)

  StringRef getArgument() const override { return "RzRzToRz"; }

  StringRef getDescription() const override {
    return "Collapse consecutive Rz gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    OpBuilder builder(&kernel.getBody());
    kernel.walk([&](Operation *op) { foldRzRz(op, builder); });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createRzRzToRzPass() {
  return std::make_unique<RzRzToRz>();
}
