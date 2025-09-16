#include "Support/mlir_utils.hpp"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

// Runtime includes
#include "common/RuntimeMLIR.h"

// Support includes
#include "Quake.hpp"

// Stdandard library includes
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>

namespace mqss::support::quakeDialect {
std::string getOperationName(Operation *op) {
  return op->getName().getIdentifier().getValue().str();
}

std::string getOnlyGateName(Operation *op) {
  if (!isOperatingGate(op)) {
    return "";
  }
  auto [_, gateName] = op->getName().getStringRef().split('.');
  return std::string(gateName);
}


std::tuple<ModuleOp, MLIRContext *>
extractMLIRContext(const std::string &quakeModule) {
  auto contextPtr = cudaq::initializeMLIR();
  MLIRContext &context = *contextPtr.get();

  // Get the quake representation of the kernel
  auto quakeCode = quakeModule;
  auto m_module = mlir::parseSourceString<ModuleOp>(quakeCode, &context);
  if (!m_module) {
    throw std::runtime_error("Module cannot be parsed");
  }

  return std::make_tuple(m_module.release(), contextPtr.release());
}

std::pair<ModuleOp, std::unique_ptr<MLIRContext>>
extractModuleOpAndContextPointer(const std::string &quakeModule) {
  auto contextPtr = cudaq::initializeMLIR();
  MLIRContext &context = *contextPtr.get();
  // Get the quake representation of the kernel
  auto quakeCode = quakeModule;
  auto m_module = mlir::parseSourceString<ModuleOp>(quakeCode, &context);
  if (!m_module) {
    throw std::runtime_error("Module cannot be parsed");
  }
  return std::make_pair(m_module.release(), std::move(contextPtr));
}

std::string readFileToString(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return "";
  }
  std::ostringstream fileContents;
  fileContents << file.rdbuf();
  return fileContents.str();
}

std::string getQuake(const std::string &filename) {
  return readFileToString(filename);
}


std::vector<int> getMeasurementTargets(Operation *op, int nr_qubits) {
  if (!isMeasurementGate(op)) {
    return {};
  }
  std::vector<int> targets = {};
  if (op->getOpOperands().size() != 1) {
    throw std::runtime_error("Measurement gate op is ambiguous.");
  }
  if (auto operand = op->getOpOperands().front().get();
    operand.getType().isa<quake::RefType>()) {
    targets.push_back(
        extractIndexFromQuakeExtractRefOp(operand.getDefiningOp()));
  } else if (operand.getType().isa<quake::VeqType>()) {
    // Because this function only works for a single allocation, the
    // reference to a Veq will reference all allocated qubits in the
    // range [0, nr_qubits-1].
    for (int qubitIndex = 0; qubitIndex < nr_qubits; qubitIndex++) {
      targets.push_back(qubitIndex);
    }
  } else {
    throw std::runtime_error("Measurement gate op has unsupported operand.");
  }
  return targets;
}

std::string vectorToString(const std::vector<int> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}


std::string vectorToString(const std::vector<double> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}


std::string valueRangeToString(ValueRange range) {
  std::string out;
  llvm::raw_string_ostream rso(out);
  rso << "[";
  for (size_t i = 0; i < range.size(); ++i) {
    rso << range[i];
    if (i + 1 < range.size())
      rso << ", ";
  }
  rso << "]";
  return rso.str();
}
} // namespace mqss::support::quakeDialect