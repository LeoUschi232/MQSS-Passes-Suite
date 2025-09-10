// Utils/tensor_utils.cpp
#include "Utils/tensor_utils.hpp"

// Quake + helpers
#include "Support/CodeGen/Quake.hpp"
#include "Interfaces/QASMToQuake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"

// Core MLIR
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <queue>
#include <common/RuntimeMLIR.h>

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::ModuleOp;
using mlir::Value;
using mlir::OpBuilder;
using mlir::Location;
using mlir::Operation;
using mlir::SmallVector;
using mlir::Type;
using mlir::MLIRContext;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
////////////////////////////////////////////////////////////////////////////////

namespace ai_pass_selector {

ModuleOp makeEmptyModuleWithKernel(MLIRContext &ctx,
                                   const std::string &kernelName) {
  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());

  // func @kernel() attributes {"cudaq-entrypoint","cudaq-kernel"} { return }
  auto funcType = b.getFunctionType({}, {});
  auto func = FuncOp::create(b.getUnknownLoc(), kernelName, funcType);
  func->setAttr(b.getStringAttr("cudaq-entrypoint"), b.getUnitAttr());
  func->setAttr(b.getStringAttr("cudaq-kernel"), b.getUnitAttr());
  func.addEntryBlock();
  b.setInsertionPointToEnd(&func.getBody().front());
  b.create<ReturnOp>(b.getUnknownLoc());
  module.push_back(func);
  return module;
}

Operation *findReturn(ModuleOp m) {
  Operation *ret = nullptr;
  m.walk([&](Operation *op) {
    if (llvm::isa<ReturnOp>(op))
      ret = op;
  });
  return ret;
}

std::string gateIdFor(const std::string &base,
                      int numControls,
                      bool isAdjoint) {
  // Measurements (no controls)
  if (base == "mx" || base == "my" || base == "mz")
    return base;

  // One-qubit Clifford/rotations
  if (base == "x")
    return numControls == 2 ? "ccx" : (numControls == 1 ? "cx" : "x");
  if (base == "y")
    return numControls == 1 ? "cy" : "y";
  if (base == "z")
    return numControls == 1 ? "cz" : "z";
  if (base == "h")
    return numControls == 1 ? "ch" : "h";

  if (base == "s")
    return numControls == 1
             ? (isAdjoint ? "csdg" : "cs")
             : (isAdjoint ? "sdg" : "s");
  if (base == "t")
    return numControls == 1
             ? (isAdjoint ? "ctdg" : "ct")
             : (isAdjoint ? "tdg" : "t");

  if (base == "rx")
    return numControls == 1 ? "crx" : "rx";
  if (base == "ry")
    return numControls == 1 ? "cry" : "ry";
  if (base == "rz")
    return numControls == 1 ? "crz" : "rz";

  if (base == "r1")
    return numControls == 1 ? "cp" : "r1"; // aka phase
  if (base == "u2")
    return numControls == 1 ? "cu2" : "u2";
  if (base == "u3")
    return numControls == 1 ? "cu3" : "u3";
  if (base == "u")
    return numControls == 1 ? "cu" : "u";

  if (base == "phased_rx")
    return numControls == 1 ? "cr" : "r";

  if (base == "swap")
    return numControls == 1 ? "cswap" : "swap";

  // Fallback to base (will assert in gate map if unsupported)
  return base;
}

// Turn angles[] (double) -> MLIR Value list (f64 constants).
std::vector<Value>
anglesToValues(OpBuilder &b, Location loc, llvm::ArrayRef<double> angles) {
  std::vector<Value> vals;
  vals.reserve(angles.size());
  for (double a : angles)
    vals.push_back(mqss::support::quakeDialect::createFloatValue(b, loc, a));
  return vals;
}

// Read which gate is active in an instruction row/cell.
static int activeGateIndex(const double *base) {
  for (int i = 0; i < NR_GATES; ++i)
    if (base[i] != 0.0)
      return i;
  return -1;
}

} // namespace

// ------------------------ Instruction-based ------------------------
ModuleOp ai_pass_selector::recreateQuantumCircuitFromInstructionBasedTensor(
    const InstructionBasedTensor<double> &tensor) {
  auto &ctx = *cudaq::initializeMLIR();
  ModuleOp module = makeEmptyModuleWithKernel(
      ctx, "__nvqpp__mlirgen__FromTensor");
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();

  // Insert point
  Operation *ret = findReturn(module);
  if (!ret)
    throw std::runtime_error("No return in synthesized kernel.");
  builder.setInsertionPoint(ret);

  // Infer sizes
  const int featuresPerRow = tensor.shape[1];
  const int maxQubits = (featuresPerRow - NR_GATES - MAX_GATE_PARAMS) / 2;
  const int gateOffset = 2 * maxQubits;
  const int paramOffset = gateOffset + NR_GATES;

  // Allocate single veq
  auto veqTy = quake::VeqType::get(&ctx, maxQubits);
  Value veq = builder.create<quake::AllocaOp>(loc, veqTy);

  // For each instruction
  for (int instr = 0; instr < tensor.shape[0]; ++instr) {
    const double *row = tensor.raw() + instr * featuresPerRow;

    // Skip all-zero rows (no op)
    bool any = false;
    for (int j = 0; j < featuresPerRow; ++j) {
      if (row[j] != 0.0) {
        any = true;
        break;
      }
    }
    if (!any)
      continue;

    // Controls / Targets
    std::vector<int> controlsIdx, targetsIdx;
    for (int q = 0; q < maxQubits; ++q)
      if (row[q] != 0.0)
        controlsIdx.push_back(q);
    for (int q = 0; q < maxQubits; ++q)
      if (row[maxQubits + q] != 0.0)
        targetsIdx.push_back(q);

    // Gate
    int g = activeGateIndex(row + gateOffset);
    if (g < 0)
      continue;
    auto baseGate = std::string(SUPPORTED_GATES[g]);

    // Params
    bool isAdjoint = (row[paramOffset] != 0.0);
    std::vector<double> angles;
    for (int i = 0; i < MAX_GATE_ANGLES; ++i)
      if (row[paramOffset + 1 + i] != 0.0)
        angles.push_back(row[paramOffset + 1 + i]);

    // Build refs
    std::vector<Value> controlRefs, targetRefs;
    controlRefs.reserve(controlsIdx.size());
    targetRefs.reserve(targetsIdx.size());
    for (int c : controlsIdx)
      controlRefs.push_back(
          builder.create<quake::ExtractRefOp>(loc, veq,
                                              static_cast<std::size_t>(c)));
    for (int t : targetsIdx)
      targetRefs.push_back(
          builder.create<quake::ExtractRefOp>(loc, veq,
                                              static_cast<std::size_t>(t)));
    std::vector<Value> paramVals = anglesToValues(builder, loc, angles);

    // Choose concrete gate id
    std::string gateId =
        gateIdFor(baseGate, static_cast<int>(controlRefs.size()), isAdjoint);

    // Insert op via gate map
    mqss::interfaces::insertQASMGateIntoQuakeModule(
        gateId, builder, loc, paramVals, controlRefs, targetRefs, isAdjoint);
  }

  return module;
}


// ------------------------ Depth-based ------------------------
ModuleOp ai_pass_selector::recreateQuantumCircuitFromDepthBasedTensor(
    const DepthBasedTensor<double> &tensor) {
  auto &ctx = *cudaq::initializeMLIR();
  ModuleOp module = makeEmptyModuleWithKernel(
      ctx, "__nvqpp__mlirgen__FromTensor");
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();

  Operation *ret = findReturn(module);
  if (!ret)
    throw std::runtime_error("No return in synthesized kernel.");
  builder.setInsertionPoint(ret);

  const int maxDepth = tensor.shape[0];
  const int maxQubits = tensor.shape[1];
  const int features = tensor.shape[2];

  constexpr int feature_gate_offset = 0;
  constexpr int feature_param_offset = feature_gate_offset + NR_GATES;
  constexpr int feature_control_info_offset =
      feature_param_offset + MAX_GATE_PARAMS; // [is_control, is_target]
  constexpr int feature_targets_offset =
      feature_control_info_offset + CONTROL_PARAMS;

  // Allocate single veq
  auto veqTy = quake::VeqType::get(&ctx, maxQubits);
  Value veq = builder.create<quake::AllocaOp>(loc, veqTy);

  // Scratch to avoid re-extracting the same ref many times
  std::vector refCache(maxQubits, Value{});
  auto getRef = [&](int q)-> Value {
    if (!refCache[q])
      refCache[q] = builder.create<quake::ExtractRefOp>(
          loc, veq, static_cast<std::size_t>(q));
    return refCache[q];
  };

  // Walk depth cross-sections
  for (int depth = 0; depth < maxDepth; ++depth) {
    // Collect per-qubit cells
    struct Cell {
      int gateIndex = -1;
      bool isControl = false;
      bool isTarget = false;
      bool isAdjoint = false;
      std::vector<double> angles;
      std::vector<int> linkedQubits; // controls if target, targets if control
    };
    std::vector<Cell> cells(maxQubits);

    bool anyAtThisDepth = false;
    for (int q = 0; q < maxQubits; ++q) {
      const double *base = tensor.raw() + (depth * tensor.shape[1] + q) *
                           features;
      int g = activeGateIndex(base + feature_gate_offset);
      if (g < 0)
        continue;

      anyAtThisDepth = true;
      cells[q].gateIndex = g;
      cells[q].isControl = base[feature_control_info_offset + 0] != 0.0;
      cells[q].isTarget = base[feature_control_info_offset + 1] != 0.0;
      cells[q].isAdjoint = base[feature_param_offset + 0] != 0.0;

      for (int i = 0; i < MAX_GATE_ANGLES; ++i) {
        if (double v = base[feature_param_offset + 1 + i]; v != 0.0)
          cells[q].angles.push_back(v);
      }
      for (int t = 0; t < maxQubits; ++t) {
        if (base[feature_targets_offset + t] != 0.0)
          cells[q].linkedQubits.push_back(t);
      }
    }
    if (!anyAtThisDepth)
      continue;

    // Group qubits by gate index
    std::unordered_map<int, std::vector<int> > qubitsByGate;
    for (int q = 0; q < maxQubits; ++q)
      if (cells[q].gateIndex >= 0)
        qubitsByGate[cells[q].gateIndex].push_back(q);

    // For each gate kind, build instances
    for (auto &[gate, members] : qubitsByGate) {
      const auto baseGate = std::string(SUPPORTED_GATES[gate]);

      // Build a bipartite linkage using the symmetric link arrays.
      // We'll create connected components over controls+targets.
      std::vector<char> visited(maxQubits, 0);

      for (int seed : members) {
        if (visited[seed])
          continue;

        // BFS over this gate’s subgraph
        std::vector<int> component;
        std::queue<int> q;
        q.push(seed);
        visited[seed] = 1;
        while (!q.empty()) {
          int u = q.front();
          q.pop();
          component.push_back(u);
          // traverse symmetric links
          for (int v : cells[u].linkedQubits) {
            if (cells[v].gateIndex != gate)
              continue; // only within same gate kind
            if (!visited[v]) {
              visited[v] = 1;
              q.push(v);
            }
          }
          // Also add peers (same gate kind at this depth) that point back to u
          for (int v : members) {
            if (!visited[v] && cells[v].gateIndex == gate) {
              if (std::find(cells[v].linkedQubits.begin(),
                            cells[v].linkedQubits.end(), u)
                  != cells[v].linkedQubits.end()) {
                visited[v] = 1;
                q.push(v);
              }
            }
          }
        }

        // Split component into controls / targets
        std::vector<int> controls, targets;
        for (int v : component) {
          if (cells[v].isControl)
            controls.push_back(v);
          if (cells[v].isTarget)
            targets.push_back(v);
        }

        // Params / adjoint (take from any member)
        bool isAdjoint = false;
        std::vector<double> angles;
        for (int v : component) {
          isAdjoint = cells[v].isAdjoint;
          if (!cells[v].angles.empty()) {
            angles = cells[v].angles;
            break;
          }
        }

        // Special-case: two-target, no-control gates like SWAP
        if (baseGate == "swap" && controls.empty() && targets.size() == 2) {
          std::vector<Value> emptyParams, emptyCtrls;
          std::vector tRefs = {getRef(targets[0]),
                               getRef(targets[1])};
          mqss::interfaces::insertQASMGateIntoQuakeModule(
              "swap", builder, loc, emptyParams, emptyCtrls, tRefs, false);
          continue;
        }

        // Single-qubit ops (no controls, 1 target)
        if (controls.empty() && targets.size() == 1) {
          std::string gateId = gateIdFor(baseGate, 0, isAdjoint);
          auto params = anglesToValues(builder, loc, angles);
          std::vector<Value> emptyCtrls, tRefs = {getRef(targets[0])};
          mqss::interfaces::insertQASMGateIntoQuakeModule(
              gateId, builder, loc, params, emptyCtrls, tRefs, isAdjoint);
          continue;
        }

        // Controlled ops (support common 1c/1t, and some known 2c/1t X)
        std::string gateId = gateIdFor(baseGate,
                                       static_cast<int>(controls.size()),
                                       isAdjoint);
        auto params = anglesToValues(builder, loc, angles);

        // If multiple targets exist (e.g., separate CXs at same depth),
        // emit an op per (control-set, target) pair using the symmetric links.
        if (!controls.empty() && !targets.empty()) {
          // Build quick membership set for controls per target using links.
          for (int t : targets) {
            // collect controls that point to this target
            std::vector<Value> ctrlsForThisTarget;
            for (int c : controls) {
              bool linked = std::find(cells[c].linkedQubits.begin(),
                                      cells[c].linkedQubits.end(), t)
                            != cells[c].linkedQubits.end();
              // also accept symmetric link from target back to control
              linked = linked || std::find(cells[t].linkedQubits.begin(),
                                           cells[t].linkedQubits.end(), c)
                       != cells[t].linkedQubits.end();
              if (linked)
                ctrlsForThisTarget.push_back(getRef(c));
            }
            if (ctrlsForThisTarget.empty())
              continue;

            std::vector tRef = {getRef(t)};
            mqss::interfaces::insertQASMGateIntoQuakeModule(
                gateId, builder, loc, params, ctrlsForThisTarget, tRef,
                isAdjoint);
          }
          continue;
        }

        // Measurements (mx/my/mz) on multiple qubits in same layer
        if ((baseGate == "mx" || baseGate == "my" || baseGate == "mz")
            && controls.empty() && !targets.empty()) {
          std::string gate_id = baseGate;
          std::vector<Value> emptyParams, emptyCtrls, tRefs;
          for (int t : targets)
            tRefs.push_back(getRef(t));
          mqss::interfaces::insertQASMGateIntoQuakeModule(
              gate_id, builder, loc, emptyParams, emptyCtrls, tRefs, false);
          continue;
        }

        // Fallback: emit one op with all targets and all controls (best-effort)
        {
          std::vector<Value> ctrlRefs, tgtRefs;
          for (int c : controls)
            ctrlRefs.push_back(getRef(c));
          for (int t : targets)
            tgtRefs.push_back(getRef(t));
          mqss::interfaces::insertQASMGateIntoQuakeModule(
              gateId, builder, loc, params, ctrlRefs, tgtRefs, isAdjoint);
        }
      } // components
    } // per gate kind
  } // depth

  return module;
}