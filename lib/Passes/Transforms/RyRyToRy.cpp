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
#define GEN_PASS_DEF_RYRYTORY
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {

void foldRyRy(Operation *op, OpBuilder &builder) {
  auto ry2 = dyn_cast_or_null<quake::RyOp>(*op);
  if (!ry2 || !ry2.getControls().empty() || ry2.getTargets().size() != 1)
    return;
  auto prev =
      supportQuake::getPreviousOperationOnTarget(ry2, ry2.getTargets()[0]);
  if (!prev)
    return;
  auto ry1 = dyn_cast_or_null<quake::RyOp>(prev);
  if (!ry1 || !ry1.getControls().empty() || ry1.getTargets().size() != 1)
    return;
  builder.setInsertionPoint(ry2);
  auto p1 = supportQuake::getParametersValues(ry1.getParameters());
  auto p2 = supportQuake::getParametersValues(ry2.getParameters());
  if (p1.size() != p2.size())
    return;
  SmallVector<Value> params;
  for (size_t i = 0; i < p1.size(); ++i) {
    params.push_back(
        supportQuake::createFloatValue(builder, ry2.getLoc(), p1[i] + p2[i]));
  }
  mlir::IRRewriter rewriter(ry2->getContext());
  rewriter.setInsertionPointAfter(ry2);
  rewriter.create<quake::RyOp>(ry2.getLoc(), ry2.isAdj(), params,
                               ry2.getControls(), ry2.getTargets());
  rewriter.eraseOp(ry2);
  rewriter.eraseOp(ry1);
}

class RyRyToRy : public BaseMQSSPass<RyRyToRy> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RyRyToRy)

  llvm::StringRef getArgument() const override { return "RyRyToRy"; }

  llvm::StringRef getDescription() const override {
    return "Collapse consecutive Ry gates";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    OpBuilder builder(&kernel.getBody());
    kernel.walk([&](Operation *op) { foldRyRy(op, builder); });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createRyRyToRyPass() {
  return std::make_unique<RyRyToRy>();
}
