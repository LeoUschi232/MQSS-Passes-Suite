#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

// Environment includes
#include "Environment/quantum_circuit_tensor.hpp"

// MLIR includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::Location;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::Value;
using mlir::func::FuncOp;
////////////////////////////////////////////////////////////////////////////////

namespace ai_pass_selector {

/**
 *
 * @param ctx
 * @param kernelName
 * @return
 */
ModuleOp makeEmptyModuleWithKernel(MLIRContext &ctx,
                                   const std::string &kernelName);

/**
 *
 * @param m
 * @return
 */
Operation *findReturn(ModuleOp m);

/**
 *
 * @param base
 * @param numControls
 * @param isAdjoint
 * @return
 */
std::string gateIdFor(const std::string &base, int numControls, bool isAdjoint);

/**
 *
 * @param b
 * @param loc
 * @param angles
 * @return
 */
std::vector<Value> anglesToValues(OpBuilder &b, Location loc,
                                  llvm::ArrayRef<double> angles);

/**
 *
 * @param base
 * @return
 */
static int activeGateIndex(const double *base);

/**
 *
 * @param tensor
 * @return
 */
ModuleOp recreateQuantumCircuitFromInstructionBasedTensor(
    MLIRContext &ctx, const InstructionBasedTensor<double> &tensor);

/**
 *
 * @param tensor
 * @return
 */
ModuleOp recreateQuantumCircuitFromDepthBasedTensor(
    MLIRContext &ctx, const DepthBasedTensor<double> &tensor);

} // namespace ai_pass_selector

#endif // TENSOR_UTILS_H