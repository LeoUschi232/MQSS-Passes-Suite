#include "Utils/tensor_utils.hpp"
// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"
// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Location.h"
#include "cudaq/Support/Plugin.h"
#include "common/RuntimeMLIR.h"
// Support includes
#include "Quake.hpp"
// Standard library includes
#include <memory>
#include <algorithm>
#include <cmath>

using mlir::ModuleOp;
using mlir::UnknownLoc;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::SmallVector;
using mlir::Block;
using mlir::Type;
using mlir::IntegerAttr;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
ModuleOp recreateQuantumCircuitFromInstructionBasedTensor(
    const InstructionBasedTensor<double> &tensor) {
  auto context_ptr = cudaq::initializeMLIR();
  MLIRContext *context = context_ptr.get();
  ModuleOp module = ModuleOp::create(UnknownLoc::get(context));
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(module.getBody());
  int feature_size = tensor.shape[1];
  int max_q = (feature_size - NR_GATES - MAX_GATE_PARAMS) / 2;
  // First pass: determine num_qubits and num_bits
  int num_qubits = 0;
  int num_bits = 0;
  for (int r = 0; r < tensor.shape[0]; ++r) {
    const double *row = tensor.raw() + static_cast<std::size_t>(r) * feature_size;
    int gate_index = -1;
    for (int g = 0; g < NR_GATES; ++g) {
      if (std::abs(row[2 * max_q + g] - 1.0) < 1e-6) {
        gate_index = g;
        break;
      }
    }
    if (gate_index == -1) continue;
    std::string_view gate = SUPPORTED_GATES[gate_index];
    for (int c = 0; c < max_q; ++c) {
      if (std::abs(row[c] - 1.0) < 1e-6) {
        num_qubits = std::max(num_qubits, c + 1);
      }
    }
    int tgt_count = 0;
    for (int t = 0; t < max_q; ++t) {
      if (std::abs(row[max_q + t] - 1.0) < 1e-6) {
        num_qubits = std::max(num_qubits, t + 1);
        tgt_count++;
      }
    }
    if (gate == "mx" || gate == "my" || gate == "mz") {
      num_bits += tgt_count;
    }
  }
  // Create func
  SmallVector<Type, 4> result_types(num_bits, builder.getI1Type());
  auto func_type = builder.getFunctionType({}, result_types);
  auto loc = module.getLoc();
  auto func = builder.create<FuncOp>(loc, "circuit", func_type);
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  // Alloca
  auto ref_type = quake::RefType::get(context);
  auto veq_type = quake::VeqType::get(context, num_qubits);
  Value veq = builder.create<quake::AllocaOp>(loc, veq_type);
  // Second pass: create ops
  std::vector<Value> measure_results;
  for (int r = 0; r < tensor.shape[0]; ++r) {
    const double *row = tensor.raw() + static_cast<std::size_t>(r) * feature_size;
    int gate_index = -1;
    for (int g = 0; g < NR_GATES; ++g) {
      if (std::abs(row[2 * max_q + g] - 1.0) < 1e-6) {
        gate_index = g;
        break;
      }
    }
    if (gate_index == -1) continue;
    std::string_view gate = SUPPORTED_GATES[gate_index];
    bool adj = std::abs(row[2 * max_q + NR_GATES] - 1.0) < 1e-6;
    // Params
    std::vector<Value> param_ops;
    int num_params = 0;
    if (gate == "rx" || gate == "ry" || gate == "rz" || gate == "r1") num_params = 1;
    else if (gate == "u2" || gate == "phased_rx") num_params = 2;
    else if (gate == "u3") num_params = 3;
    for (int p = 0; p < num_params; ++p) {
      double v = row[2 * max_q + NR_GATES + 1 + p];
      param_ops.push_back(createFloatValue(builder, loc, v));
    }
    // Controls
    std::vector<Value> control_refs;
    for (int c = 0; c < max_q; ++c) {
      if (std::abs(row[c] - 1.0) < 1e-6) {
        IntegerAttr raw_index = builder.getIntegerAttr(builder.getIndexType(), c);
        Value ref = builder.create<quake::ExtractRefOp>(loc, veq, raw_index);
        control_refs.push_back(ref);
      }
    }
    // Targets
    std::vector<Value> target_refs;
    for (int t = 0; t < max_q; ++t) {
      if (std::abs(row[max_q + t] - 1.0) < 1e-6) {
        IntegerAttr raw_index = builder.getIntegerAttr(builder.getIndexType(), t);
        Value ref = builder.create<quake::ExtractRefOp>(loc, veq, raw_index);
        target_refs.push_back(ref);
      }
    }
    // Validate target count
    if (gate == "swap" && target_refs.size() != 2) {
      throw std::runtime_error("Swap gate requires exactly two target qubits");
    }
    if ((gate != "swap" && gate != "mx" && gate != "my" && gate != "mz") && target_refs.size() != 1) {
      throw std::runtime_error("Non-swap, non-measurement gate requires exactly one target qubit");
    }
    // Create op
    if (gate == "x") {
      builder.create<quake::XOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gate == "y") {
      builder.create<quake::YOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gate == "z") {
      builder.create<quake::ZOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gate == "h") {
      builder.create<quake::HOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gate == "s") {
      builder.create<quake::SOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gate == "t") {
      builder.create<quake::TOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gate == "rx") {
      builder.create<quake::RxOp>(loc, adj, control_refs, param_ops[0], target_refs[0]);
    } else if (gate == "ry") {
      builder.create<quake::RyOp>(loc, adj, control_refs, param_ops[0], target_refs[0]);
    } else if (gate == "rz") {
      builder.create<quake::RzOp>(loc, adj, control_refs, param_ops[0], target_refs[0]);
    } else if (gate == "swap") {
      builder.create<quake::SwapOp>(loc, adj, control_refs, target_refs[0], target_refs[1]);
    } else if (gate == "r1") {
      builder.create<quake::R1Op>(loc, adj, control_refs, param_ops[0], target_refs[0]);
    } else if (gate == "u2") {
      builder.create<quake::U2Op>(loc, adj, control_refs, param_ops[0], param_ops[1], target_refs[0]);
    } else if (gate == "u3") {
      builder.create<quake::U3Op>(loc, adj, control_refs, param_ops[0], param_ops[1], param_ops[2], target_refs[0]);
    } else if (gate == "phased_rx") {
      builder.create<quake::PhasedRxOp>(loc, adj, control_refs, param_ops[0], param_ops[1], target_refs[0]);
    } else if (gate == "mx") {
      auto op = builder.create<quake::MxOp>(loc, builder.getI1Type(), adj, control_refs, target_refs[0]);
      measure_results.push_back(op.getResult(0));
    } else if (gate == "my") {
      auto op = builder.create<quake::MyOp>(loc, builder.getI1Type(), adj, control_refs, target_refs[0]);
      measure_results.push_back(op.getResult(0));
    } else if (gate == "mz") {
      auto op = builder.create<quake::MzOp>(loc, builder.getI1Type(), adj, control_refs, target_refs[0]);
      measure_results.push_back(op.getResult(0));
    }
  }
  // Return
  builder.create<ReturnOp>(loc, measure_results);
  return module;
}

ModuleOp recreateQuantumCircuitFromDepthBasedTensor(
    const DepthBasedTensor<double> &tensor) {
  constexpr int feature_gate_offset = 0;
  constexpr int feature_param_offset = feature_gate_offset + NR_GATES;
  constexpr int feature_control_info_offset = feature_param_offset + MAX_GATE_PARAMS;
  constexpr int feature_targets_offset = feature_control_info_offset + CONTROL_PARAMS;
  auto context_ptr = cudaq::initializeMLIR();
  MLIRContext *context = context_ptr.get();
  ModuleOp module = ModuleOp::create(UnknownLoc::get(context));
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(module.getBody());
  int max_depth = tensor.shape[0];
  int max_q = tensor.shape[1];
  int feature_size = tensor.shape[2];
  // First pass: determine num_qubits and num_bits, collect gates
  int num_qubits = 0;
  int num_bits = 0;
  struct GateInfo {
    std::string_view gate;
    bool adj;
    std::vector<double> angles;
    std::vector<int> controls;
    std::vector<int> targets;
  };
  std::vector<GateInfo> gates;
  for (int d = 0; d < max_depth; ++d) {
    std::vector<bool> visited(max_q, false);
    for (int q = 0; q < max_q; ++q) {
      if (visited[q]) continue;
      const double *cell = tensor.raw() + (static_cast<std::size_t>(d) * max_q + q) * feature_size;
      int gate_index = -1;
      for (int g = 0; g < NR_GATES; ++g) {
        if (std::abs(cell[feature_gate_offset + g] - 1.0) < 1e-6) {
          gate_index = g;
          break;
        }
      }
      if (gate_index == -1) continue;
      std::string_view gate = SUPPORTED_GATES[gate_index];
      bool is_control = std::abs(cell[feature_control_info_offset] - 1.0) < 1e-6;
      bool is_target = std::abs(cell[feature_control_info_offset + 1] - 1.0) < 1e-6;
      bool adj = std::abs(cell[feature_param_offset] - 1.0) < 1e-6;
      std::vector<double> angles(MAX_GATE_ANGLES, 0.0);
      for (int p = 0; p < MAX_GATE_ANGLES; ++p) {
        angles[p] = cell[feature_param_offset + 1 + p];
      }
      std::vector<int> connected;
      for (int t = 0; t < max_q; ++t) {
        if (std::abs(cell[feature_targets_offset + t] - 1.0) < 1e-6) {
          connected.push_back(t);
        }
      }
      GateInfo gi;
      gi.gate = gate;
      gi.adj = adj;
      gi.angles = angles;
      if (is_control) {
        gi.controls.push_back(q);
        gi.targets = connected;
      } else if (is_target) {
        gi.targets.push_back(q);
        gi.controls = connected;
      } else {
        continue; // Invalid
      }
      // Mark visited
      visited[q] = true;
      for (int o : connected) {
        visited[o] = true;
      }
      // Update num_qubits
      for (int c : gi.controls) num_qubits = std::max(num_qubits, c + 1);
      for (int t : gi.targets) num_qubits = std::max(num_qubits, t + 1);
      // Update num_bits
      if (gate == "mx" || gate == "my" || gate == "mz") {
        num_bits += static_cast<int>(gi.targets.size());
      }
      gates.push_back(gi);
    }
  }
  // Create func
  SmallVector<Type, 4> result_types(num_bits, builder.getI1Type());
  auto func_type = builder.getFunctionType({}, result_types);
  auto loc = module.getLoc();
  FuncOp func = builder.create<FuncOp>(loc, "circuit", func_type);
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  // Alloca
  auto ref_type = quake::RefType::get(context);
  auto veq_type = quake::VeqType::get(context, num_qubits);
  Value veq = builder.create<quake::AllocaOp>(loc, veq_type);
  // Create ops
  std::vector<Value> measure_results;
  for (const auto &gi : gates) {
    bool adj = gi.adj;
    // Params
    std::vector<Value> param_ops;
    int num_params = 0;
    if (gi.gate == "rx" || gi.gate == "ry" || gi.gate == "rz" || gi.gate == "r1") num_params = 1;
    else if (gi.gate == "u2" || gi.gate == "phased_rx") num_params = 2;
    else if (gi.gate == "u3") num_params = 3;
    for (int p = 0; p < num_params; ++p) {
      param_ops.push_back(createFloatValue(builder, loc, gi.angles[p]));
    }
    // Controls
    std::vector<Value> control_refs;
    for (int c : gi.controls) {
      IntegerAttr raw_index = builder.getIntegerAttr(builder.getIndexType(), c);
      Value ref = builder.create<quake::ExtractRefOp>(loc, veq, raw_index);
      control_refs.push_back(ref);
    }
    // Targets
    std::vector<Value> target_refs;
    for (int t : gi.targets) {
      IntegerAttr raw_index = builder.getIntegerAttr(builder.getIndexType(), t);
      Value ref = builder.create<quake::ExtractRefOp>(loc, veq, raw_index);
      target_refs.push_back(ref);
    }
    // Validate target count
    if (gi.gate == "swap" && target_refs.size() != 2) {
      throw std::runtime_error("Swap gate requires exactly two target qubits");
    }
    if ((gi.gate != "swap" && gi.gate != "mx" && gi.gate != "my" && gi.gate != "mz") && target_refs.size() != 1) {
      throw std::runtime_error("Non-swap, non-measurement gate requires exactly one target qubit");
    }
    // Create op
    if (gi.gate == "x") {
      builder.create<quake::XOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gi.gate == "y") {
      builder.create<quake::YOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gi.gate == "z") {
      builder.create<quake::ZOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gi.gate == "h") {
      builder.create<quake::HOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gi.gate == "s") {
      builder.create<quake::SOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gi.gate == "t") {
      builder.create<quake::TOp>(loc, adj, control_refs, target_refs[0]);
    } else if (gi.gate == "rx") {
      builder.create<quake::RxOp>(loc, adj, control_refs, param_ops[0], target_refs[0]);
    } else if (gi.gate == "ry") {
      builder.create<quake::RyOp>(loc, adj, control_refs, param_ops[0], target_refs[0]);
    } else if (gi.gate == "rz") {
      builder.create<quake::RzOp>(loc, adj, control_refs, param_ops[0], target_refs[0]);
    } else if (gi.gate == "swap") {
      builder.create<quake::SwapOp>(loc, adj, control_refs, target_refs[0], target_refs[1]);
    } else if (gi.gate == "r1") {
      builder.create<quake::R1Op>(loc, adj, control_refs, param_ops[0], target_refs[0]);
    } else if (gi.gate == "u2") {
      builder.create<quake::U2Op>(loc, adj, control_refs, param_ops[0], param_ops[1], target_refs[0]);
    } else if (gi.gate == "u3") {
      builder.create<quake::U3Op>(loc, adj, control_refs, param_ops[0], param_ops[1], param_ops[2], target_refs[0]);
    } else if (gi.gate == "phased_rx") {
      builder.create<quake::PhasedRxOp>(loc, adj, control_refs, param_ops[0], param_ops[1], target_refs[0]);
    } else if (gi.gate == "mx") {
      auto op = builder.create<quake::MxOp>(loc, builder.getI1Type(), adj, control_refs, target_refs[0]);
      measure_results.push_back(op.getResult(0));
    } else if (gi.gate == "my") {
      auto op = builder.create<quake::MyOp>(loc, builder.getI1Type(), adj, control_refs, target_refs[0]);
      measure_results.push_back(op.getResult(0));
    } else if (gi.gate == "mz") {
      auto op = builder.create<quake::MzOp>(loc, builder.getI1Type(), adj, control_refs, target_refs[0]);
      measure_results.push_back(op.getResult(0));
    }
  }
  // Return
  builder.create<ReturnOp>(loc, measure_results);
  return module;
}
} // namespace ai_pass_selector