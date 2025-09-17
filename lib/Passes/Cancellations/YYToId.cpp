#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_YYTOID
// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class YYToId final : public BaseMQSSPass<YYToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(YYToId)

  StringRef getArgument() const override { return "YYToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive Y gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      patternCancellation<quake::YOp, quake::YOp>(op, 0, 1, 0, 1);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createYYToIdPass() {
  return std::make_unique<YYToId>();
}