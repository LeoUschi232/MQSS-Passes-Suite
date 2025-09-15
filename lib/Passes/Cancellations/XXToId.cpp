#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_XXTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class XXToId final : public BaseMQSSPass<XXToId> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XXToId)

  StringRef getArgument() const override { return "XXToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive X gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      patternCancellation<quake::XOp, quake::XOp>(op, 0, 1, 0, 1);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createXXToIdPass() {
  return std::make_unique<XXToId>();
}