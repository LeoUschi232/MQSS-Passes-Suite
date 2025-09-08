#include "Utils/mlir_utils.hpp"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

// Cudaq includes
#include "cudaq/Frontend/nvqpp/AttributeNames.h"

// Runtime includes
#include "common/RuntimeMLIR.h"

// Stdandard library includes
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>

namespace ai_pass_selector {


std::string getOperationName(mlir::Operation *op) {
  return op->getName().getIdentifier().getValue().str();
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

std::string getQuake(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return "";
  }
  std::ostringstream fileContents;
  fileContents << file.rdbuf();
  return fileContents.str();
}
} // namespace ai_pass_selector