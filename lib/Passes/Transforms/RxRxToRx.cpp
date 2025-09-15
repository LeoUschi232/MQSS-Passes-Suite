#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_RXRXTORX
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {

void foldRxRx(Operation *op, OpBuilder &builder) {
  auto rx2 = dyn_cast_or_null<quake::RxOp>(*op);
  if (!rx2 || !rx2.getControls().empty() || rx2.getTargets().size() != 1)
    return;
  auto prev =
      supportQuake::getPreviousOperationOnTarget(rx2, rx2.getTargets()[0]);
  if (!prev)
    return;
  auto rx1 = dyn_cast_or_null<quake::RxOp>(prev);
  if (!rx1 || !rx1.getControls().empty() || rx1.getTargets().size() != 1)
    return;
  builder.setInsertionPoint(rx2);
  auto p1 = supportQuake::getParametersValues(rx1.getParameters());
  auto p2 = supportQuake::getParametersValues(rx2.getParameters());
  if (p1.size() != p2.size())
    return;
  SmallVector<Value> params;
  for (size_t i = 0; i < p1.size(); ++i) {
    params.push_back(
        supportQuake::createFloatValue(builder, rx2.getLoc(), p1[i] + p2[i]));
  }
  IRRewriter rewriter(rx2->getContext());
  rewriter.setInsertionPointAfter(rx2);
  rewriter.create<quake::RxOp>(rx2.getLoc(), rx2.isAdj(), params,
                               rx2.getControls(), rx2.getTargets());
  rewriter.eraseOp(rx2);
  rewriter.eraseOp(rx1);
}

class RxRxToRx final : public BaseMQSSPass<RxRxToRx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RxRxToRx)

  StringRef getArgument() const override { return "RxRxToRx"; }

  StringRef getDescription() const override {
    return "Collapse consecutive Rx gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    OpBuilder builder(&kernel.getBody());
    kernel.walk([&](Operation *op) { foldRxRx(op, builder); });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createRxRxToRxPass() {
  return std::make_unique<RxRxToRx>();
}
