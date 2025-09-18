#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HCRXHTOCRZ

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class HCrxHToCrz final : public BaseMQSSPass<HCrxHToCrz> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCrxHToCrz)

  StringRef getArgument() const override { return "HCrxHToCrz"; }

  StringRef getDescription() const override {
    return "Fold H CRx H to CRz";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    kernel.walk([&](Operation *op) {
      // TODO: Implement the actual transformation logic here.
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHCrxHToCrzPass() {
  return std::make_unique<HCrxHToCrz>();
}