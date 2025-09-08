#include "PassSelector/environment.hpp"

//  Quake includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

// Passes includes
#include "Passes/Cancellations.hpp"

// Support includes
#include "Support/CodeGen/Quake.hpp"

// AI Utils includes
#include "Utils/mlir_utils.hpp"

// Stdandard library includes
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


using namespace mlir;
using namespace mqss::opt;
using namespace mqss::support::quakeDialect;
namespace fs = std::filesystem;

namespace ai_pass_selector {
int indent;

struct IdentRAII {
  int &indent;

  IdentRAII(int &indent) : indent(indent) {
  }

  ~IdentRAII() { --indent; }
};

void resetIndent() { indent = 0; }
IdentRAII pushIndent() { return IdentRAII(++indent); }

llvm::raw_ostream &printIndent() {
  for (int i = 0; i < indent; ++i)
    llvm::outs() << "  ";
  return llvm::outs();
}

void printBlock(Block &block) {
  // Print the block intrinsics properties (basically: argument list)
  printIndent() << "Block with " << block.getNumArguments() << " arguments, "
      << block.getNumSuccessors()
      << " successors, and "
      // Note, this `.size()` is traversing a linked-list and is O(n).
      << block.getOperations().size() << " operations\n";

  // Block main role is to hold a list of Operations: let's recurse.
  auto indent = pushIndent();
  for (Operation &op : block.getOperations())
    printOperation(&op);
}

void printRegion(Region &region) {
  // A region does not hold anything by itself other than a list of blocks.
  printIndent() << "Region with " << region.getBlocks().size() << " blocks:\n";
  auto indent = pushIndent();
  for (Block &block : region.getBlocks())
    printBlock(block);
}

std::string vecToString(std::vector<int> v) {
  const std::string res = accumulate(
      v.begin(), v.end(), std::string(), [](std::string s, int n) {
        return s + (s.empty() ? "" : ",") + std::to_string(n);
      });
  return "[" + res + "]";
}

std::string vecToString(std::vector<double> v) {
  const std::string res = accumulate(
      v.begin(), v.end(), std::string(), [](std::string s, double n) {
        return s + (s.empty() ? "" : ",") + std::to_string(n);
      });
  return "[" + res + "]";
}

int printOperation(Operation *op) {

  if (op->getDialect()->getNamespace() != "quake" || isa<quake::AllocaOp>(op) ||
      isa<quake::ExtractRefOp>(op)) {
    return 0;
  }
  auto gateName = std::string(op->getName().getStringRef());
  if (const size_t pos = gateName.find("quake."); pos != std::string::npos) {
    gateName.erase(pos, 6); // 6 is the length of "quake."
  }
  std::vector<int> measurements;
  bool isAdj = false;
  if (!(isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op))) {
    auto gate = dyn_cast<quake::OperatorInterface>(op);
    std::vector<double> parameters = getParametersValues(gate.getParameters());
    std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
    std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
    isAdj = gate.isAdj();
    std::cout << gateName << ": " << vecToString(parameters) << ", " <<
        vecToString(targets) << ", " << vecToString(controls) << ", " << isAdj
        << std::endl;
  }
  return 1;
}


void f1(const std::string &subdirectory) {
  const fs::path quake_dir = fs::path(AI_DATASET_DIR) / "Quake" / subdirectory;
  if (!fs::exists(quake_dir)) {
    std::cerr << "Input directory does not exist: " << quake_dir.string()
        << std::endl;
    return;
  }
  for (const auto &entry : fs::directory_iterator(quake_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".qke") {
      fs::path filename = quake_dir / (entry.path().stem().string() + ".qke");

      // The getQuake function just reads in the string contents of the file.
      std::string quakeModule = getQuake(filename.string());
      auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
      MLIRContext &context = *contextPtr;
      mlirModule.walk(
          [&](Operation *op) { printOperation(op); });
    }
  }
}

} // namespace ai_pass_selector