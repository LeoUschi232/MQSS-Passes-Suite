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

std::string getOperationName(Operation *op) {
  return op->getName().getIdentifier().getValue().str();
}

std::string getOnlyGateName(Operation *op) {
  if (!mqss::support::quakeDialect::isOperatingGate(op)) {
    return "";
  }
  auto [_, gateName] = op->getName().getStringRef().split('.');
  return std::string(gateName);
}


std::tuple<mlir::ModuleOp, mlir::MLIRContext *>
extractMLIRContext(const std::string &quakeModule) {
  auto contextPtr = cudaq::initializeMLIR();
  mlir::MLIRContext &context = *contextPtr.get();

  // Get the quake representation of the kernel
  auto quakeCode = quakeModule;
  auto m_module = mlir::parseSourceString<mlir::ModuleOp>(quakeCode, &context);
  if (!m_module) {
    throw std::runtime_error("Module cannot be parsed");
  }

  return std::make_tuple(m_module.release(), contextPtr.release());
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