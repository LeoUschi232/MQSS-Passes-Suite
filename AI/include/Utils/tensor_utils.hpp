#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::ModuleOp;
using mlir::func::FuncOp;
////////////////////////////////////////////////////////////////////////////////

namespace ai_pass_selector {

/**
 *
 * @param tensor
 * @return
 */
ModuleOp recreateQuantumCircuitFromInstructionBasedTensor(
const InstructionBasedTensor<double> &tensor);

/**
 *
 * @param tensor
 * @return
 */
ModuleOp recreateQuantumCircuitFromDepthBasedTensor(
    const DepthBasedTensor<double> &tensor);

} // namespace ai_pass_selector

#endif // TENSOR_UTILS_H