/* Auto-generated simple pass implementing YH -> HY transformation */
#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_YHTOHY
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class YHToHY : public BaseMQSSPass<YHToHY> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(YHToHY)
  llvm::StringRef getArgument() const override { return "YHToHY"; }
  llvm::StringRef getDescription() const override {
    return "Pass that switches a pattern composed by Y and Hadamard to "
           "Hadamard and Y";
  }
  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      patternSwitch<quake::YOp, quake::HOp, quake::HOp, quake::YOp>(op);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createYHToHYPass() {
  return std::make_unique<YHToHY>();
}
