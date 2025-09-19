// Utils/tensor_utils.cpp
#include "Utils/tensor_utils.hpp"

#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "common/RuntimeMLIR.h"

#include <iostream>

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::ModuleOp;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::Location;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::Value;
using mlir::ValueRange;
////////////////////////////////////////////////////////////////////////////////
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

// void insertMeasurements(
//     RebuildSetup &rebuildSetup, int gateIndex,
//     const std::vector<int> &targets) {
//   if (targets.empty()) {
//     throw std::runtime_error("Measurement needs at least 1 target.");
//   }
//   std::map<int, std::pair<std::string, int> > classicalRegToQVector = {};
//
//   for (int target : targets) {
//     llvm::SmallVector<Value> targetValues  = {
//       rebuildSetup.builder.create<quake::ExtractRefOp>(
//         rebuildSetup.loc, rebuildSetup.getRef(target), target)
//     };
//     mlir::Type measTy = quake::MeasureType::get(rebuildSetup.builder.getContext());
//   }
//   switch (gateIndex) {
//   case MX:
//     rebuildSetup.builder.create<quake::MxOp>(
//         rebuildSetup.loc, targetsVec);
//     break;
//   case MY:
//     rebuildSetup.builder.create<quake::MyOp>(
//         rebuildSetup.loc, targetsVec);
//     break;
//   case MZ:
//     rebuildSetup.builder.create<quake::MzOp>(
//         rebuildSetup.loc, targetsVec);
//     break;
//   default:
//     throw std::runtime_error("Not a measurement: "
//                              + std::string(SUPPORTED_GATES[gateIndex]));
//   }
// }

void insertMeasurements(
    RebuildSetup &rebuildSetup, int gateIndex, ValueRange targets) {
  if (targets.empty()) {
    throw std::runtime_error("Measurement needs at least 1 target.");
  }
  llvm::SmallVector<Value> targetsVec(targets.begin(), targets.end());
  mlir::Type measureType = quake::MeasureType::get(
      rebuildSetup.builder.getContext());
  switch (gateIndex) {
  case MX:
    rebuildSetup.builder.create<quake::MxOp>(
        rebuildSetup.loc, measureType, targetsVec);
    break;
  case MY:
    rebuildSetup.builder.create<quake::MyOp>(
        rebuildSetup.loc, measureType, targetsVec);
    break;
  case MZ:
    rebuildSetup.builder.create<quake::MzOp>(
        rebuildSetup.loc, measureType, targetsVec);
    break;
  default:
    throw std::runtime_error(
        "Not a measurement: " + std::string(SUPPORTED_GATES[gateIndex]));
  }
}

void insertGate(
    RebuildSetup &rebuildSetup, int gateIndex, bool isAdj,
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
    rebuildSetup.builder.create<quake::XOp>(
        rebuildSetup.loc, false,
        ValueRange{}, controls, targets);
    break;
  case Y:
    rebuildSetup.builder.create<quake::YOp>(
        rebuildSetup.loc, false,
        ValueRange{}, controls, targets);
    break;
  case Z:
    rebuildSetup.builder.create<quake::ZOp>(
        rebuildSetup.loc, false,
        ValueRange{}, controls, targets);
    break;
  case H:
    rebuildSetup.builder.create<quake::HOp>(
        rebuildSetup.loc, false,
        ValueRange{}, controls, targets);
    break;
  case S:
    rebuildSetup.builder.create<quake::SOp>(
        rebuildSetup.loc, isAdj,
        ValueRange{}, controls, targets);
    break;
  case T:
    rebuildSetup.builder.create<quake::TOp>(
        rebuildSetup.loc, isAdj,
        ValueRange{}, controls, targets);
    break;
  case RX:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0])
    };
    rebuildSetup.builder.create<quake::RxOp>(
        rebuildSetup.loc, isAdj,
        ValueRange(paramsVector), controls, targets);
    break;
  case RY:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0])
    };
    rebuildSetup.builder.create<quake::RyOp>(
        rebuildSetup.loc, isAdj,
        ValueRange(paramsVector), controls, targets);
    break;
  case RZ:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0])
    };
    rebuildSetup.builder.create<quake::RzOp>(
        rebuildSetup.loc, isAdj,
        ValueRange(paramsVector), controls, targets);
    break;
  case SWAP:
    if (targets.size() != 2) {
      throw std::runtime_error("Swap requires exactly 2 targets.");
    }
    rebuildSetup.builder.create<quake::SwapOp>(
        rebuildSetup.loc, false,
        ValueRange{}, controls, targets);
    break;
  case R1:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0])
    };
    rebuildSetup.builder.create<quake::R1Op>(
        rebuildSetup.loc, isAdj,
        ValueRange(paramsVector), controls, targets);
    break;
  case U2:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0]),
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[1])
    };
    rebuildSetup.builder.create<quake::U2Op>(
        rebuildSetup.loc, isAdj,
        ValueRange(paramsVector), controls, targets);
    break;
  case U3:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0]),
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[1]),
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[2])
    };
    rebuildSetup.builder.create<quake::U3Op>(
        rebuildSetup.loc, isAdj,
        ValueRange(paramsVector), controls, targets);
    break;
  case PHASED_RX:
    paramsVector = {
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[0]),
        createFloatValue(rebuildSetup.builder, rebuildSetup.loc, angles[1])
    };
    rebuildSetup.builder.create<quake::PhasedRxOp>(
        rebuildSetup.loc, isAdj,
        ValueRange(paramsVector), controls, targets);
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
    throw std::runtime_error(
        "Unsupported gate index: " + std::to_string(gateIndex));
  }
}


// ----------------- High-level “with-context” wrappers -----------------
std::pair<ModuleOp, std::unique_ptr<MLIRContext> >
recreateQuantumCircuitFromInstructionBasedTensorWithContext(
    const InstructionBasedTensor<double> &tensor) {
  // Decide maxQubits from tensor shape
  const int maxInstructions = tensor.shape[0];
  const int featuresPerRow = tensor.shape[1];
  const int maxQubits = featuresPerRow - NR_GATES - MAX_GATE_PARAMS;

  auto rebuildSetup = beginReconstruction(
      "__nvqpp__mlirgen__FromTensor", maxQubits);

  for (int instr = 0; instr < maxInstructions; instr++) {
    std::vector<int> controlIndexes, targetIndexes;
    int j = 0;
    for (; j < maxQubits; j++) {
      if (double value = tensor(instr, j); value == -1.0) {
        controlIndexes.push_back(j);
      } else if (value == 1.0) {
        targetIndexes.push_back(j);
      } else if (value != 0.0) {
        throw std::runtime_error("Control trigger: " + std::to_string(value));
      }
    }
    if (targetIndexes.empty()) {
      // Empty targets means the circuit reached its end.
      // All further rows MUST be empty too.
      break;
    }

    int gateIndex = -1;
    bool isAdj = false;
    for (; j < maxQubits + NR_GATES; j++) {
      if (double value = tensor(instr, j); std::abs(value) == 1.0) {
        if (gateIndex >= 0) {
          throw std::runtime_error("Multiple gates triggered in one row.");
        }
        gateIndex = j - maxQubits;
        if (value < 0.0) {
          isAdj = true;
        }
      } else if (value != 0.0) {
        throw std::runtime_error("Gate trigger: " + std::to_string(value));
      }
    }
    if (gateIndex < 0) {
      // Empty instruction means the circuit reached its end.
      // All further rows MUST be empty too.
      break;
    }

    std::vector<double> angles;
    for (; j < featuresPerRow; j++) {
      angles.push_back(tensor(instr, j));
    }
    if (angles.size() != MAX_GATE_PARAMS) {
      throw std::runtime_error(
          "Nr gate angles: " + std::to_string(angles.size()));
    }
    insertGate(rebuildSetup, gateIndex, isAdj,
               controlIndexes, targetIndexes, angles);
  }
  return {rebuildSetup.module, std::move(rebuildSetup.ctxOwner)};
}

std::pair<ModuleOp, std::unique_ptr<MLIRContext> >
recreateQuantumCircuitFromDepthBasedTensorWithContext(
    const DepthBasedTensor<double> &tensor) {
  const int maxDepth = tensor.shape[0];
  const int maxQubits = tensor.shape[1];

  auto rebuildSetup = beginReconstruction(
      "__nvqpp__mlirgen__FromTensor", maxQubits);

  for (int depth = 0; depth < maxDepth; depth++) {
    std::vector qubitsHandled(maxQubits, false);

    for (int qubit = 0; qubit < maxQubits; qubit++) {
      if (qubitsHandled[qubit]) {
        continue;
      }

      int j = 0;
      int gateIndex = -1;
      bool isAdj = false;
      for (; j < NR_GATES; j++) {
        if (double value = tensor(depth, qubit, j); std::abs(value) == 1.0) {
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
        angles.push_back(tensor(depth, qubit, j));
      }
      if (angles.size() != MAX_GATE_PARAMS) {
        throw std::runtime_error(
            "Nr gate angles: " + std::to_string(angles.size()));
      }
      bool isControl = false;
      bool isTarget = false;
      if (double value = tensor(depth, qubit, j++); value == -1.0) {
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
           extraQubit < maxQubits && j < rolesBase + maxQubits;
           extraQubit++, j++) {
        if (j >= rolesBase + maxQubits) {
          throw std::runtime_error("Iterator exceeded qubit roles.");
        }
        if (double value = tensor(depth, qubit, j); value == -1.0) {
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

      insertGate(rebuildSetup, gateIndex, isAdj,
                 controlIndexes, targetIndexes, angles);
    }
  }
  return {rebuildSetup.module, std::move(rebuildSetup.ctxOwner)};
}


} // namespace ai_pass_selector