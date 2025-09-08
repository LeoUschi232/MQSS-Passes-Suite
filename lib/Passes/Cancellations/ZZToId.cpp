#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mqss::opt {
#define GEN_PASS_DEF_ZZTOID
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class ZZToId : public BaseMQSSPass<ZZToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZZToId)

  llvm::StringRef getArgument() const override { return "ZZToId"; }

  llvm::StringRef getDescription() const override {
    return "Remove consecutive Z gates.";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      patternCancellation<quake::ZOp, quake::ZOp>(op, 0, 1, 0, 1);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZZToIdPass() {
  return std::make_unique<ZZToId>();
}
