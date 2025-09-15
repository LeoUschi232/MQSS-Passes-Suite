/* Auto-generated simple pass implementing HY -> YH transformation */
#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HYTOYH

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class HYToYH final : public BaseMQSSPass<HYToYH> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HYToYH)
  StringRef getArgument() const override { return "HYToYH"; }
  StringRef getDescription() const override {
    return "Pass that switches a pattern composed Hadamard and Y to Y and "
           "Hadamard";
  }
  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      patternSwitch<quake::HOp, quake::YOp, quake::YOp, quake::HOp>(op);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHYToYHPass() {
  return std::make_unique<HYToYH>();
}
