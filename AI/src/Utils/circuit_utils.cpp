#include "Utils/circuit_utils.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"

namespace ai_pass_selector {

unsigned int classify_circuit(
    unsigned int nr_qubits,
    unsigned int nr_instructions,
    unsigned int depth) {
  if (nr_qubits <= TINY_CIRCUIT_SPECS[0]
      && nr_instructions <= TINY_CIRCUIT_SPECS[1]
      && depth <= TINY_CIRCUIT_SPECS[2]) {
    return TINY;
  }
  if (nr_qubits <= SMALL_CIRCUIT_SPECS[0]
      && nr_instructions <= SMALL_CIRCUIT_SPECS[1]
      && depth <= SMALL_CIRCUIT_SPECS[2]) {
    return SMALL;
  }
  if (nr_qubits <= MODERATE_CIRCUIT_SPECS[0]
      && nr_instructions <= MODERATE_CIRCUIT_SPECS[1]
      && depth <= MODERATE_CIRCUIT_SPECS[2]) {
    return MODERATE;
  }
  if (nr_qubits <= BIG_CIRCUIT_SPECS[0]
      && nr_instructions <= BIG_CIRCUIT_SPECS[1]
      && depth <= BIG_CIRCUIT_SPECS[2]) {
    return BIG;
  }
  return HUGE;
}

std::vector<std::string> split_string(const std::string &s, char delim) {
  std::vector<std::string> parts;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    parts.push_back(item);
  }
  return parts;
}

FuncOp getKernelEntryPoint(ModuleOp module) {
  if (!module) {
    return nullptr;
  }

  FuncOp fallback_kernel;
  for (FuncOp func : module.getOps<FuncOp>()) {
    if (func->hasAttr("cudaq-entrypoint")) {
      return func;
    }
    if (!fallback_kernel && func->hasAttr("cudaq-kernel")) {
      fallback_kernel = func;
    }
  }

  if (fallback_kernel) {
    return fallback_kernel;
  }

  module.emitError("Failed to locate a cudaq kernel entry function.");
  return nullptr;
}

} // namespace ai_pass_selector
