#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "Support/Transforms/CommutateOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CXZTOZCX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class CxZToZCx final : public BaseMQSSPass<CxZToZCx> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxZToZCx)

  StringRef getArgument() const override { return "CxZToZCx"; }

  StringRef getDescription() const override {
    return "Apply commutation pass of the pattern CNot-Z to Z-CNot";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      auto cxOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1) {
        return;
      }
      auto optional_zOp
          = getNextOperationOnTarget(cxOp, cxOp.getControls()[0]);
      if (!optional_zOp) {
        return;
      }
      auto zOp = dyn_cast_or_null<quake::ZOp>(*optional_zOp);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(cxOp->getContext());
      rewriter.setInsertionPointAfter(zOp);
      ValueRange targets = cxOp.getTargets();
      ValueRange controls = cxOp.getControls();
      Location loc = cxOp.getLoc();
      rewriter.create<quake::ZOp>(loc, false, controls);
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.eraseOp(cxOp);
      rewriter.eraseOp(zOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxZToZCxPass() {
  return std::make_unique<CxZToZCx>();
}