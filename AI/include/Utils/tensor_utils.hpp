#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

// Environment includes
#include "Environment/quantum_circuit.hpp"
#include "Environment/quantum_circuit_tensor.hpp"

// MLIR includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using mlir::Location;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::FuncOp;

namespace ai_pass_selector {
constexpr int X = GATE_INDEX("x"sv);
constexpr int Y = GATE_INDEX("y"sv);
constexpr int Z = GATE_INDEX("z"sv);
constexpr int H = GATE_INDEX("h"sv);
constexpr int S = GATE_INDEX("s"sv);
constexpr int T = GATE_INDEX("t"sv);
constexpr int RX = GATE_INDEX("rx"sv);
constexpr int RY = GATE_INDEX("ry"sv);
constexpr int RZ = GATE_INDEX("rz"sv);
constexpr int SWAP = GATE_INDEX("swap"sv);
constexpr int R1 = GATE_INDEX("r1"sv);
constexpr int U2 = GATE_INDEX("u2"sv);
constexpr int U3 = GATE_INDEX("u3"sv);
constexpr int PHASED_RX = GATE_INDEX("phased_rx"sv);
constexpr int MX = GATE_INDEX("mx"sv);
constexpr int MY = GATE_INDEX("my"sv);
constexpr int MZ = GATE_INDEX("mz"sv);

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
                                  llvm::ArrayRef<double> angles);

/**
 *
 * @param rebuildSetup
 * @param gateIndex
 * @param targets
 */
static void insertMeasurements(RebuildSetup &rebuildSetup, int gateIndex,
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
void insertGate(RebuildSetup &rebuildSetup, int gateIndex,
                const std::vector<int> &targetIndexes,
                const std::vector<int> &controlIndexes = {},
                const std::vector<double> &angles = {}, bool isAdj = false);

/**
 * Count which qubits are actually used in the instruction-based circuit tensor.
 * @param tensor
 * @return
 */
unsigned int nrUsedQubitsInTensor(const InstructionsTensor<double> &tensor);

/**
 *
 * @param tensor
 * @return
 */
QuantumCircuit
recreateQuantumCircuitFromTensor(const InstructionsTensor<double> &tensor);

} // namespace ai_pass_selector

#endif // TENSOR_UTILS_HPP
