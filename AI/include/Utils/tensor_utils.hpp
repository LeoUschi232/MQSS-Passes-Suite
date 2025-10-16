#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

// Environment includes
#include "Environment/quantum_circuit.hpp"
#include "Environment/quantum_circuit_tensor.hpp"

// MLIR includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

// Standard library includes
#include <string_view>

using namespace std::literals;

using mlir::Location;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::FuncOp;

namespace ai_pass_selector {
enum class GateSymbol : int {
  X = GATE_INDEX("x"sv),
  Y = GATE_INDEX("y"sv),
  Z = GATE_INDEX("z"sv),
  H = GATE_INDEX("h"sv),
  S = GATE_INDEX("s"sv),
  T = GATE_INDEX("t"sv),
  RX = GATE_INDEX("rx"sv),
  RY = GATE_INDEX("ry"sv),
  RZ = GATE_INDEX("rz"sv),
  SWAP = GATE_INDEX("swap"sv),
  R1 = GATE_INDEX("r1"sv),
  U2 = GATE_INDEX("u2"sv),
  U3 = GATE_INDEX("u3"sv),
  PHASED_RX = GATE_INDEX("phased_rx"sv),
  MX = GATE_INDEX("mx"sv),
  MY = GATE_INDEX("my"sv),
  MZ = GATE_INDEX("mz"sv)
};

constexpr GateSymbol GATE_SYMBOL(std::string_view gate) {
  for (int i = 0; i < NR_GATES; i++) {
    if (SUPPORTED_GATES[i] == gate) {
      return static_cast<GateSymbol>(i);
    }
  }
  throw std::invalid_argument("GATE_SYMBOL: unsupported gate " +
                              std::string(gate));
}

constexpr int to_gate_index(GateSymbol symbol) {
  return static_cast<int>(symbol);
}

struct RebuildSetup {
  std::unique_ptr<MLIRContext> ctxOwner;
  OpBuilder builder;
  Location loc;
  ModuleOp module;
  FuncOp entry;
  Value veq;
  std::vector<Value> refCache;

  // Move-only to avoid accidental copies of MLIR handles.
  RebuildSetup(const RebuildSetup &) = delete;

  RebuildSetup &operator=(const RebuildSetup &) = delete;

  RebuildSetup(RebuildSetup &&) noexcept = default;

  RebuildSetup &operator=(RebuildSetup &&) noexcept = default;

  explicit RebuildSetup(std::unique_ptr<MLIRContext> ctxOwnerIn)
      : ctxOwner(std::move(ctxOwnerIn)), builder(ctxOwner.get()),
        loc(builder.getUnknownLoc()), module(ModuleOp::create(loc)) {}

  Value getRef(int idx);

  std::vector<Value> getRefs(const std::vector<int> &indexes);
};

/**
 *
 * @param depths
 * @return
 */
unsigned int get_max_depth(std::vector<unsigned int> depths);

/**
 * Create fresh context + empty module + @kernel + alloca veq.
 * Insertion point is set *before* return so you can emit ops.
 * @param kernelName
 * @param maxQubits
 * @return
 */
RebuildSetup beginQuantumCircuitConstruction(const std::string &kernelName,
                                             int maxQubits);

/**
 * Utility: find the func.return inside a module (in case you need it)
 * @param module
 * @return
 */
Operation *findReturn(ModuleOp module);

/**
 * Angles[] (double) → MLIR f64 constants.
 * @param builder
 * @param loc
 * @param angles
 * @return
 */
std::vector<Value> anglesToValues(OpBuilder &builder, Location loc,
                                  llvm::ArrayRef<float> angles);

/**
 *
 * @param rebuildSetup
 * @param gateIndex
 * @param targets
 */
static void insertMeasurements(RebuildSetup &rebuildSetup, GateSymbol gate,
                               ValueRange targets);

/**
 *
 * @param rebuildSetup
 * @param gateIndex
 * @param targetIndexes
 * @param controlIndexes
 * @param angles
 * @param isAdj
 */
void insertGate(RebuildSetup &rebuildSetup, GateSymbol gate,
                const std::vector<int> &targetIndexes,
                const std::vector<int> &controlIndexes = {},
                const std::vector<float> &angles = {}, bool isAdj = false);

/**
 * Count which qubits are actually used in the instruction-based circuit tensor.
 * @param tensor
 * @return
 */
unsigned int nrUsedQubitsInTensor(const InstructionsTensor<float> &tensor);

/**
 *
 * @param tensor
 * @return
 */
QuantumCircuit
recreateQuantumCircuitFromTensor(const InstructionsTensor<float> &tensor);

/**
 *
 * @param tensor
 */
void check_tensor(const InstructionsTensor<float> &tensor);

/**
 *
 * @param tensor
 * @return
 */
std::string tensor_to_string(const torch::Tensor &tensor);
} // namespace ai_pass_selector

#endif // TENSOR_UTILS_HPP
