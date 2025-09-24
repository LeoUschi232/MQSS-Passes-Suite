// Utils/tensor_utils.cpp
#include "Utils/tensor_utils.hpp"

// Support includes
#include "Support/mlir_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using mlir::Location;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
////////////////////////////////////////////////////////////////////////////////

// Cudaq includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

// MLIR includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

// Common includes
#include "common/RuntimeMLIR.h"

// Standard library includes
#include <iostream>

using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {

// ----------------- tiny internal helpers -----------------
static FuncOp makeEmptyKernel(OpBuilder &b, ModuleOp m,
                              const std::string &kernelName) {
  auto funcType = b.getFunctionType({}, {});
  auto func = FuncOp::create(b.getUnknownLoc(), kernelName, funcType);
  func->setAttr(b.getStringAttr("cudaq-entrypoint"), b.getUnitAttr());
  func->setAttr(b.getStringAttr("cudaq-kernel"), b.getUnitAttr());
  func.addEntryBlock();
  b.setInsertionPointToEnd(&func.getBody().front());
  b.create<ReturnOp>(b.getUnknownLoc());
  m.push_back(func);
  return func;
}

Operation *findReturn(ModuleOp module) {
  Operation *ret = nullptr;
  module.walk([&](Operation *op) {
    if (llvm::isa<ReturnOp>(op))
      ret = op;
  });
  return ret;
}

std::vector<Value> anglesToValues(OpBuilder &builder, Location loc,
                                  const std::vector<double> &angles) {
  std::vector<Value> vals;
  vals.reserve(angles.size());
  for (double angle : angles) {
    vals.push_back(createFloatValue(builder, loc, angle));
  }
  return vals;
}

// ----------------- RebuildSetup API -----------------
RebuildSetup beginReconstruction(const std::string &kernelName, int maxQubits) {
  RebuildSetup setup(cudaq::initializeMLIR());
  auto *ctx = setup.ctxOwner.get();
  OpBuilder &builder = setup.builder;

  // Create @kernel and place insertion before return
  setup.entry = makeEmptyKernel(builder, setup.module, kernelName);
  if (Operation *ret = findReturn(setup.module)) {
    builder.setInsertionPoint(ret);
  } else {
    throw std::runtime_error("No return in synthesized kernel.");
  }

  // Allocate one veq
  auto veqTy = quake::VeqType::get(ctx, maxQubits);
  setup.veq = builder.create<quake::AllocaOp>(setup.loc, veqTy);

  // Prepare ref cache
  setup.refCache.assign(maxQubits, Value{});
  return setup;
}

Value RebuildSetup::getRef(int idx) {
  if (idx < 0 || idx >= static_cast<int>(refCache.size())) {
    llvm::report_fatal_error("getRef index out of range");
  }
  if (!refCache[idx]) {
    refCache[idx] = builder.create<quake::ExtractRefOp>(
        loc, veq, static_cast<std::size_t>(idx));
  }
  return refCache[idx];
}

std::vector<Value> RebuildSetup::getRefs(const std::vector<int> &indexes) {
  if (indexes.size() == 0) {
    return {};
  }
  std::vector<Value> refsVector;
  for (int idx : indexes) {
    if (const Value ref = getRef(idx); ref) {
      refsVector.push_back(ref);
    }
  }
  return refsVector;
}

void insertMeasurements(RebuildSetup &rebuildSetup, int gateIndex,
                        ValueRange targets) {
  if (targets.empty()) {
    throw std::runtime_error("Measurement needs at least 1 target.");
  }
  llvm::SmallVector<Value> targetsVec(targets.begin(), targets.end());
  mlir::Type measureType =
      quake::MeasureType::get(rebuildSetup.builder.getContext());
  switch (gateIndex) {
  case MX:
    rebuildSetup.builder.create<quake::MxOp>(rebuildSetup.loc, measureType,
                                             targetsVec);
    break;
  case MY:
    rebuildSetup.builder.create<quake::MyOp>(rebuildSetup.loc, measureType,
                                             targetsVec);
    break;
  case MZ:
    rebuildSetup.builder.create<quake::MzOp>(rebuildSetup.loc, measureType,
                                             targetsVec);
    break;
  default:
    throw std::runtime_error("Not a measurement: " +
                             std::string(SUPPORTED_GATES[gateIndex]));
  }
}

void insertGate(RebuildSetup &rebuildSetup, int gateIndex, bool isAdj,
                const std::vector<int> &controlIndexes,
                const std::vector<int> &targetIndexes,
                const std::vector<double> &angles) {
  // HOLD the storage for the whole iteration.
  std::vector<Value> controlVals = rebuildSetup.getRefs(controlIndexes);
  std::vector<Value> targetVals = rebuildSetup.getRefs(targetIndexes);

  // Create ranges that view the storage above.
  ValueRange controls(controlVals);
  ValueRange targets(targetVals);
  std::vector<Value> paramsVector;
  switch (gateIndex) {
  case X:
    rebuildSetup.builder.create<quake::XOp>(rebuildSetup.loc, false,
                                            ValueRange{}, controls, targets);
    break;
  case Y:
    rebuildSetup.builder.create<quake::YOp>(rebuildSetup.loc, false,
                                            ValueRange{}, controls, targets);
    break;
  case Z:
    rebuildSetup.builder.create<quake::ZOp>(rebuildSetup.loc, false,
                                            ValueRange{}, controls, targets);
    break;
  case H:
    rebuildSetup.builder.create<quake::HOp>(rebuildSetup.loc, false,
                                            ValueRange{}, controls, targets);
    break;
  case S:
    rebuildSetup.builder.create<quake::SOp>(rebuildSetup.loc, isAdj,
                                            ValueRange{}, controls, targets);
    break;
  case T:
    rebuildSetup.builder.create<quake::TOp>(rebuildSetup.loc, isAdj,
                                            ValueRange{}, controls, targets);
    break;
  case RX:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0])};
    rebuildSetup.builder.create<quake::RxOp>(
        rebuildSetup.loc, isAdj, ValueRange(paramsVector), controls, targets);
    break;
  case RY:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0])};
    rebuildSetup.builder.create<quake::RyOp>(
        rebuildSetup.loc, isAdj, ValueRange(paramsVector), controls, targets);
    break;
  case RZ:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0])};
    rebuildSetup.builder.create<quake::RzOp>(
        rebuildSetup.loc, isAdj, ValueRange(paramsVector), controls, targets);
    break;
  case SWAP:
    if (targets.size() != 2) {
      throw std::runtime_error("Swap requires exactly 2 targets.");
    }
    rebuildSetup.builder.create<quake::SwapOp>(rebuildSetup.loc, false,
                                               ValueRange{}, controls, targets);
    break;
  case R1:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0])};
    rebuildSetup.builder.create<quake::R1Op>(
        rebuildSetup.loc, isAdj, ValueRange(paramsVector), controls, targets);
    break;
  case U2:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0]),
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[1])};
    rebuildSetup.builder.create<quake::U2Op>(
        rebuildSetup.loc, isAdj, ValueRange(paramsVector), controls, targets);
    break;
  case U3:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0]),
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[1]),
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[2])};
    rebuildSetup.builder.create<quake::U3Op>(
        rebuildSetup.loc, isAdj, ValueRange(paramsVector), controls, targets);
    break;
  case PHASED_RX:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0]),
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[1])};
    rebuildSetup.builder.create<quake::PhasedRxOp>(
        rebuildSetup.loc, isAdj, ValueRange(paramsVector), controls, targets);
    break;
  case MX:
    if (!controls.empty()) {
      throw std::runtime_error("Mx cannot have controls.");
    }
    insertMeasurements(rebuildSetup, MX, targets);
    break;
  case MY:
    if (!controls.empty()) {
      throw std::runtime_error("My cannot have controls.");
    }
    insertMeasurements(rebuildSetup, MY, targets);
    break;
  case MZ:
    if (!controls.empty()) {
      throw std::runtime_error("Mz cannot have controls.");
    }
    insertMeasurements(rebuildSetup, MZ, targets);
    break;
  default:
    throw std::runtime_error("Unsupported gate index: " +
                             std::to_string(gateIndex));
  }
}

unsigned int
nrUsedQubitsInInstructionBasedTensor(const InstructionsTensor<double> &tensor) {
  const int nr_instructions = tensor.shape[0];
  const int instruction_features = tensor.shape[1];
  const int max_qubits = instruction_features - NR_GATES - MAX_GATE_PARAMS;
  unsigned int max_used_qubit_index = 0;
  for (unsigned int instr = 0; instr < nr_instructions; instr++) {
    for (unsigned int j = 0; j < max_qubits; j++) {
      if (double value = tensor(instr, j); value == -1.0) {
        max_used_qubit_index = std::max(max_used_qubit_index, j);
      } else if (value == 1.0) {
        max_used_qubit_index = std::max(max_used_qubit_index, j);
      } else if (value != 0.0) {
        throw std::runtime_error("Control trigger: " + std::to_string(value));
      }
    }
  }
  return max_used_qubit_index + 1;
}

// ----------------- High-level “with-context” wrappers -----------------
std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
recreateQuantumCircuitFromInstructionBasedTensorWithContext(
    const InstructionsTensor<double> &tensor) {
  // Decide maxQubits from tensor shape
  const int nr_instructions = tensor.shape[0];
  const int instruction_features = tensor.shape[1];
  const int max_qubits = instruction_features - NR_GATES - MAX_GATE_PARAMS;
  const int nr_qubits = nrUsedQubitsInInstructionBasedTensor(tensor);

  auto rebuildSetup =
      beginReconstruction("__nvqpp__mlirgen__FromTensor", nr_qubits);

  for (int instr = 0; instr < nr_instructions; instr++) {
    std::vector<int> controlIndexes, targetIndexes;
    int j = 0;
    for (; j < nr_qubits; j++) {
      if (double value = tensor(instr, j); value == -1.0) {
        controlIndexes.push_back(j);
      } else if (value == 1.0) {
        targetIndexes.push_back(j);
      } else if (value != 0.0) {
        throw std::runtime_error("Control trigger: " + std::to_string(value));
      }
    }
    // Skip unused qubits.
    j = max_qubits;
    if (targetIndexes.empty()) {
      std::cerr << "Warning: Found instruction without targets in "
                   "InstructionsTensor."
                << std::endl;
      // Empty targets means the circuit reached its end.
      // All further rows MUST be empty too.
      break;
    }

    int gateIndex = -1;
    bool isAdj = false;
    for (; j < max_qubits + NR_GATES; j++) {
      if (double value = tensor(instr, j); std::abs(value) == 1.0) {
        if (gateIndex >= 0) {
          throw std::runtime_error("Multiple gates triggered in one row.");
        }
        gateIndex = j - max_qubits;
        if (value < 0.0) {
          isAdj = true;
        }
      } else if (value != 0.0) {
        throw std::runtime_error("Gate trigger: " + std::to_string(value));
      }
    }
    if (gateIndex < 0) {
      std::cerr << "Warning: Found instruction without gate in "
                   "InstructionsTensor."
                << std::endl;
      // Empty instruction means the circuit reached its end.
      // All further rows MUST be empty too.
      break;
    }

    std::vector<double> angles;
    for (; j < instruction_features; j++) {
      angles.push_back(tensor(instr, j));
    }
    if (angles.size() != MAX_GATE_PARAMS) {
      throw std::runtime_error("Nr gate angles: " +
                               std::to_string(angles.size()));
    }
    insertGate(rebuildSetup, gateIndex, isAdj, controlIndexes, targetIndexes,
               angles);
  }
  return {rebuildSetup.module, std::move(rebuildSetup.ctxOwner)};
}

std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
recreateQuantumCircuitFromDepthBasedTensorWithContext(
    const DepthsTensor<double> &tensor) {
  const int depth = tensor.shape[0];
  const int max_qubits = tensor.shape[1];
  const int nr_qubits = nrUsedQubitsInDepthBasedTensor(tensor);

  auto rebuildSetup =
      beginReconstruction("__nvqpp__mlirgen__FromTensor", nr_qubits);

  for (int layer = 0; layer < depth; layer++) {
    std::vector qubitsHandled(nr_qubits, false);

    for (int qubit = 0; qubit < nr_qubits; qubit++) {
      if (qubitsHandled[qubit]) {
        continue;
      }

      int j = 0;
      int gateIndex = -1;
      bool isAdj = false;
      for (; j < NR_GATES; j++) {
        if (double value = tensor(layer, qubit, j); std::abs(value) == 1.0) {
          if (gateIndex >= 0) {
            throw std::runtime_error("Multiple gates triggered in one cell.");
          }
          gateIndex = j;
          if (value < 0.0) {
            isAdj = true;
          }
        } else if (value != 0.0) {
          throw std::runtime_error("Gate trigger: " + std::to_string(value));
        }
      }
      if (gateIndex < 0) {
        // There will probably be a lot of empty cells.
        continue;
      }

      std::vector<double> angles;
      for (; j < NR_GATES + MAX_GATE_PARAMS; j++) {
        angles.push_back(tensor(layer, qubit, j));
      }
      if (angles.size() != MAX_GATE_PARAMS) {
        throw std::runtime_error("Nr gate angles: " +
                                 std::to_string(angles.size()));
      }
      bool isControl = false;
      bool isTarget = false;
      if (double value = tensor(layer, qubit, j++); value == -1.0) {
        isControl = true;
      } else if (value == 1.0) {
        isTarget = true;
      } else if (value != 0.0) {
        throw std::runtime_error("IsControl trigger: " + std::to_string(value));
      }
      if (!isControl && !isTarget) {
        throw std::runtime_error("Qubit must be either control or target.");
      }

      constexpr int rolesBase = NR_GATES + MAX_GATE_PARAMS + QUBIT_ROLE;
      std::vector<int> controlIndexes;
      std::vector<int> targetIndexes;
      for (int extraQubit = 0;
           extraQubit < max_qubits && j < rolesBase + max_qubits;
           extraQubit++, j++) {
        if (j >= rolesBase + max_qubits) {
          throw std::runtime_error("Iterator exceeded qubit roles.");
        }
        if (double value = tensor(layer, qubit, j); value == -1.0) {
          controlIndexes.push_back(extraQubit);
        } else if (value == 1.0) {
          targetIndexes.push_back(extraQubit);
        } else if (value != 0.0) {
          throw std::runtime_error("Role trigger: " + std::to_string(value));
        }
      }
      if (targetIndexes.empty()) {
        // There will probably be a lot of empty targets.
        continue;
      }

      for (int control : controlIndexes) {
        qubitsHandled[control] = true;
      }
      for (int target : targetIndexes) {
        qubitsHandled[target] = true;
      }
      insertGate(rebuildSetup, gateIndex, isAdj, controlIndexes, targetIndexes,
                 angles);
    }
  }
  return {rebuildSetup.module, std::move(rebuildSetup.ctxOwner)};
}

} // namespace ai_pass_selector