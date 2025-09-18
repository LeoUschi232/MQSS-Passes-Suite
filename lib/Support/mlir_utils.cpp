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

std::pair<ModuleOp, std::unique_ptr<MLIRContext *> >
extractModuleOpAndContextPointer(const std::string &quakeModule) {
  auto contextPtr = cudaq::initializeMLIR();
  MLIRContext &context = *contextPtr.get();
  // Get the quake representation of the kernel
  auto quakeCode = quakeModule;
  auto m_module = mlir::parseSourceString<ModuleOp>(quakeCode, &context);
  if (!m_module) {
    throw std::runtime_error("Module cannot be parsed");
  }
  return {m_module.release(),
          std::make_unique<MLIRContext *>(contextPtr.release())};
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
    auto targetIndexOpt
        = extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
    if (!targetIndexOpt.has_value()) {
      return {};
    }
    targets.push_back(targetIndexOpt.value());
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


std::tuple<unsigned int, unsigned int, unsigned int>
getQubitsInstructionsDepth(FuncOp circuit) {
  unsigned int nrQubits = 0;
  unsigned int nrGates = 0;
  std::vector<unsigned int> depths;

  circuit.walk([&](Operation *op) {
    if (isa<quake::AllocaOp>(op)) {
      if (auto allocOp = dyn_cast<quake::AllocaOp>(op);
        allocOp.getType().dyn_cast<quake::RefType>()) {
        nrQubits += 1;
      } else if (auto qvecType = allocOp.getType().dyn_cast<quake::VeqType>()) {
        nrQubits += qvecType.getSize();
      }
      depths.resize(nrQubits, 0);
      return;
    }
    if (!isOperatingGate(op)) {
      return;
    }
    if (isMeasurementGate(op)) {
      for (auto operand : op->getOperands()) {
        if (operand.getType().isa<quake::RefType>()) {
          auto qubitIndexOpt
              = extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
          if (!qubitIndexOpt.has_value()) {
            continue;
          }
          int qubitIndex = qubitIndexOpt.value();
          if (0 <= qubitIndex && qubitIndex < nrQubits) {
            nrGates++;
            depths[qubitIndex]++;
          }
        } else if (operand.getType().isa<quake::VeqType>()) {
          // Because this function only works for a single allocation, the
          // reference to a Veq will reference all allocated qubits in the
          // range [0, nrQubits-1].
          for (int qubitIndex = 0; qubitIndex < nrQubits; qubitIndex++) {
            depths[qubitIndex]++;
          }
          nrGates += operand.getType().dyn_cast<quake::VeqType>().getSize();
        }
      }
    } else {
      nrGates++;
      auto gate = dyn_cast<quake::OperatorInterface>(op);
      std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
      std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
      targets.insert(targets.end(), controls.begin(), controls.end());
      unsigned int max_depth = 0;
      for (int qubit : targets) {
        max_depth = std::max(max_depth, depths[qubit]);
      }
      for (int qubit : targets) {
        depths[qubit] = max_depth + 1;
      }
    }
  });
  return {nrQubits, nrGates, *std::ranges::max_element(depths)};
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