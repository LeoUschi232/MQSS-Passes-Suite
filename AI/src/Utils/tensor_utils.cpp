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
unsigned int get_max_depth(std::vector<unsigned int> depths) {
  return depths.empty() ? 0 : *std::max_element(depths.begin(), depths.end());
}

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
RebuildSetup beginQuantumCircuitConstruction(const std::string &kernelName,
                                             int maxQubits) {
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

void insertGate(RebuildSetup &rebuildSetup, int gateIndex,
                const std::vector<int> &targetIndexes,
                const std::vector<int> &controlIndexes,
                const std::vector<double> &angles, bool isAdj) {
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

unsigned int nrUsedQubitsInTensor(const InstructionsTensor<double> &tensor) {
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

QuantumCircuit
recreateQuantumCircuitFromTensor(const InstructionsTensor<double> &tensor) {
  const unsigned int IRP = tensor.shape[1];
  if (IRP < MIN_IRP) {
    std::cerr << "Warning: tensor has IRP smaller than minimum IRP."
              << std::endl;
    return QuantumCircuit();
  }
  const unsigned int max_qubits = IRP - NR_GATES - MAX_GATE_PARAMS;
  const unsigned int nr_qubits = nrUsedQubitsInTensor(tensor);
  const unsigned int nr_instructions = tensor.shape[0];
  unsigned int nr_gates = nr_instructions;

  auto rebuildSetup = beginQuantumCircuitConstruction(
      /*kernel_name=*/"__nvqpp__mlirgen__FromTensor", nr_qubits);
  std::vector<unsigned int> depths(nr_qubits, 0);

  for (unsigned int instr = 0u; instr < nr_instructions; instr++) {
    std::vector<int> controlIndexes, targetIndexes;
    unsigned int j = 0u;
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
      // Actual nr of gates might be smaller than the believed nr of
      // instructions extracted from the length of the tensor.
      nr_gates = instr;
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
      nr_gates = instr;
      break;
    }

    std::vector<double> angles;
    for (; j < IRP; j++) {
      angles.push_back(tensor(instr, j));
    }
    if (angles.size() != MAX_GATE_PARAMS) {
      throw std::runtime_error("Nr gate angles: " +
                               std::to_string(angles.size()));
    }
    insertGate(rebuildSetup, gateIndex, targetIndexes, controlIndexes, angles,
               isAdj);

    // Adjust the depths only if everything ran smoothly.
    std::vector<int> involvedQubits = targetIndexes;
    involvedQubits.insert(involvedQubits.end(), controlIndexes.begin(),
                          controlIndexes.end());
    unsigned int max_depth = 0;
    for (int qubit : involvedQubits) {
      max_depth = std::max(max_depth, depths[qubit]);
    }
    for (int qubit : involvedQubits) {
      depths[qubit] = max_depth + 1;
    }
  }
  return {rebuildSetup.module, std::move(rebuildSetup.ctxOwner), nr_qubits,
          nr_gates, get_max_depth(depths)};
}

} // namespace ai_pass_selector