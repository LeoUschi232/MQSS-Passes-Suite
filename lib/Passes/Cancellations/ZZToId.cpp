#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_ZZTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class ZZToId final : public BaseMQSSPass<ZZToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZZToId)

  StringRef getArgument() const override { return "ZZToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive Z gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto zOp1 = dyn_cast_or_null<quake::ZOp>(*op);
      if (!zOp1
          || !zOp1.isAdj()
          || zOp1.getTargets().size() != 1
          || !zOp1.getControls().empty()) {
        return;
      }
      auto optional_zOp2
          = getNextOperationOnTarget(zOp1, zOp1.getTargets()[0]);
      if (!optional_zOp2) {
        return;
      }
      auto zOp2 = dyn_cast_or_null<quake::ZOp>(*optional_zOp2);
      if (!zOp2
          || zOp2.isAdj()
          || zOp2.getTargets().size() != 1
          || !zOp2.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(zOp1->getContext());
      rewriter.eraseOp(zOp1);
      rewriter.eraseOp(zOp2);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZZToIdPass() {
  return std::make_unique<ZZToId>();
}