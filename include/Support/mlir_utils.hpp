#ifndef MLIR_UTILS_HPP
#define MLIR_UTILS_HPP

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

// Support includes
#include "Quake.hpp"

// Stdandard library includes
#include <string>
#include <tuple>


namespace mqss::support::quakeDialect {
/**
 * @param op The operation to extract the name from.
 * @return The name of the operations as a std::string.
 */
std::string getOperationName(Operation *op);

/**
 *
 * @param op
 * @return
 */
std::string getOnlyGateName(Operation *op);

/**
 * Extracts a MLIR module operation and the MLIR context from the quake module
 * which is the string content of a quake file.
 * @param quakeModule The string contents of the quake file.
 * @return The MLIR module and context.
 */
std::tuple<mlir::ModuleOp, mlir::MLIRContext *>
extractMLIRContext(const std::string &quakeModule);

/**
 * Extracts the string contents of a quake file.
 * @param filename The name of the quake file.
 * @return The string contents of the quake file.
 */
std::string readFileToString(const std::string &filename);

/**
 * Extracts the string contents of a quake file.
 * @param filename The name of the quake file.
 * @return The string contents of the quake file.
 */
std::string getQuake(const std::string &filename);

/**
 *
 * @param op
 * @param nr_qubits
 * @return
 */
std::vector<int> getMeasurementTargets(Operation *op, int nr_qubits);

/**
 *
 * @param op
 * @param nr_qubits
 * @return
 */
std::tuple<std::vector<int>, std::vector<int>, std::vector<double> >
getNoneMeasurementControlsTargetsParams(Operation *op, int nr_qubits);

/**
 *
 * @param vec
 * @return
 */
std::string vectorToString(const std::vector<int> &vec);

/**
 *
 * @param vec
 * @return
 */
std::string vectorToString(const std::vector<double> &vec);

/**
 *
 * @param range
 * @return
 */
std::string valueRangeToString(ValueRange range);
} // namespace mqss::support::quakeDialect
#endif // MLIR_UTILS_HPP