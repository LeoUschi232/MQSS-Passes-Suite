/* This code and any associated documentation is provided "as is"

Copyright 2025 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------
  author Martin Letras
  date   February 2025
  version 1.0
  brief
    Set of the most functions used to manipulate Quake MLIR modules. E.g.,
    contains functions to define integer, and double constants, extract
    references, get the number of qubits, classical registers, etc.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Support/CodeGen/Quake.hpp"

////////////////////////////////////////////////////////////////////////////////
/// The includes of llvm Casting must be left here before the include of cudaq
/// QuakeOps otherwise the comipler will complain that these operations do not
/// exist in the header file.
#include "llvm/Support/Casting.h"
using llvm::isa;
using llvm::cast;
using llvm::dyn_cast;
////////////////////////////////////////////////////////////////////////////////

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/SCF/IR/SCF.h"


namespace mqss::support::quakeDialect {
// Given a OpBuilder and a double value, it inserts a double in the mlir
// module pointer by the OpBuilder and returns the inserted Value
Value createFloatValue(
    OpBuilder &builder, const Location loc, const double value) {
  // Create a constant value (20.0 of type f64)
  auto valueAttr = builder.getFloatAttr(builder.getF64Type(), value);
  auto constantOp = builder.create<arith::ConstantOp>(loc, valueAttr);
  return constantOp.getResult();
}

// TODO: return -1 is not good idea
// Given an argument value as Operation, it extracts a double, it the operation
// is not double, returns -1.0 when fail
double extractDoubleArgumentValue(Operation *op) {
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op))
    if (auto floatAttr = constantOp.getValue().dyn_cast<FloatAttr>())
      return floatAttr.getValueAsDouble();
  return -1.0;
}

// TODO: return -1 is not good idea
// Given an ExtractRefOp, it extracts the integer of the index pointing that
// reference (qubit index), returns -1 when fail
int64_t
extractIndexFromQuakeExtractRefOp(Operation *op) {
  if (auto extractRefOp = llvm::dyn_cast<quake::ExtractRefOp>(op)) {
    auto rawIndexAttr =
        extractRefOp->getAttrOfType<IntegerAttr>("rawIndex");
    return rawIndexAttr.getInt();
  }
  return -1;
}

// function to get the number of qubits in a given quantum kernel
int getNumberOfQubits(func::FuncOp circuit) {
  int numQubits = 0;
  circuit.walk([&](quake::AllocaOp allocOp) {
    if (allocOp.getType().dyn_cast<quake::RefType>()) {
      numQubits += 1;
    } else if (auto qvecType = allocOp.getType().dyn_cast<quake::VeqType>()) {
      numQubits += qvecType.getSize();
    }
  });
  return numQubits;
}

int getNumberOfGates(func::FuncOp circuit) {
  if (getNumberOfQubits(circuit) == 0) {
    return 0;
  }
  int nrGates = 0;
  circuit.walk([&](Operation *op) {
    if (op->getDialect()->getNamespace() == "quake"
        && !isa<quake::AllocaOp>(op)
        && !isa<quake::ExtractRefOp>(op)) {
      nrGates++;
    }
  });
  return nrGates;
}

int getCircuitDepth(func::FuncOp circuit) {
  int nrQubits = getNumberOfQubits(circuit);
  if (nrQubits == 0) {
    return 0;
  }
  std::vector depths(nrQubits, 0);
  circuit.walk([&](Operation *op) {
    if (op->getDialect()->getNamespace() == "quake"
        && !isa<quake::AllocaOp>(op)
        && !isa<quake::ExtractRefOp>(op)) {
      if (isa<quake::MxOp>(op)
          || isa<quake::MyOp>(op)
          || isa<quake::MzOp>(op)) {
        for (auto operand : op->getOperands()) {
          // Check if it's qubit reference
          if (operand.getType().isa<quake::RefType>()) {
            int qubitIndex =
                extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
            if (0 <= qubitIndex && qubitIndex < nrQubits) {
              depths[qubitIndex]++;
            }
          } else if (operand.getType().isa<quake::VeqType>()) {
            for (int qubitIndex = 0; qubitIndex < nrQubits; qubitIndex++) {
              depths[qubitIndex]++;
            }
          }
        }
      } else {
        auto gate = dyn_cast<quake::OperatorInterface>(op);
        std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
        std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
        targets.insert(targets.end(), controls.begin(), controls.end());
        int max_depth = 0;
        for (int qubit : targets) {
          max_depth = std::max(max_depth, depths[qubit]);
        }
        for (int qubit : targets) {
          depths[qubit] = max_depth + 1;
        }
      }
    }
  });
  return *std::ranges::max_element(depths);
}

// Function to get the number of classical bits allocated in a given
// quantum kernel, it also stores information of the qubit position
int getNumberOfClassicalBits(
    func::FuncOp circuit, std::map<int, int> &measurements) {
  int numBits = 0;
  circuit.walk([&](Operation *op) {
    if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)) {
      for (auto operand : op->getOperands()) {
        // Check if it's qubit reference
        if (operand.getType().isa<quake::RefType>()) {
          int qubitIndex =
              extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
          assert(qubitIndex != -1 && "Non valid qubit index for measurement!");
          measurements[qubitIndex] = numBits;
          numBits += 1;
        } else if (operand.getType().isa<quake::VeqType>()) {
          auto qvecType = operand.getType().dyn_cast<quake::VeqType>();
          int start = numBits;
          numBits += qvecType.getSize();
          // Assume Veq qubits are sequential indices matching global bit
          // positions; may not hold for sliced/concatenated Veq
          // TODO: compute actual global indices.
          for (int i = start; i < numBits; i++) {
            measurements[i] = i;
          }
        }
      }
    }
  });
  return numBits;
}

// Function to get the number of classical bits allocated in
// a given quantum kernel
int getNumberOfClassicalBits(
    func::FuncOp circuit) {
  int numBits = 0;
  circuit.walk([&](Operation *op) {
    if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)) {
      for (auto operand : op->getOperands()) {
        if (operand.getType()
          .isa<quake::RefType>()) {
          // Check if it's a qubit reference
          int qubitIndex =
              extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
          assert(qubitIndex != -1 && "Non valid qubit index for measurement!");
          numBits += 1;
        } else if (operand.getType().isa<quake::VeqType>()) {
          auto qvecType = operand.getType().dyn_cast<quake::VeqType>();
          numBits += qvecType.getSize();
        }
      }
    }
  });
  return numBits;
}

// Function that get the indices of the Value objectes in array
std::vector<int>
getIndicesOfValueRange(const ValueRange array) {
  std::vector<int> indices;
  for (auto value : array) {
    int qubit_index = extractIndexFromQuakeExtractRefOp(value.getDefiningOp());
    indices.push_back(qubit_index);
  }
  return indices;
}

// At the moment, it is assumed that the parameters are of type Double
std::vector<double>
getParametersValues(const ValueRange array) {
  std::vector<double> parameters;
  for (auto value : array) {
    double param = extractDoubleArgumentValue(value.getDefiningOp());
    parameters.push_back(param);
  }
  return parameters;
}

// Get the previous operation on a given TargeQubit, starting from
// currentOp
Operation *getPreviousOperationOnTarget(
    Operation *currentOp, Value targetQubit) {
  // Start from the previous operation
  Operation *prevOp = currentOp->getPrevNode();
  // Iterate through the previous operations in the block
  while (prevOp) {
    // Check if the operation has a target qubit and matches the given target
    if (auto quakeOp = dyn_cast<quake::OperatorInterface>(prevOp)) {
      int targetQCurr =
          extractIndexFromQuakeExtractRefOp(targetQubit.getDefiningOp());
      for (Value target : quakeOp.getTargets()) {
        int targetQPrev =
            extractIndexFromQuakeExtractRefOp(target.getDefiningOp());
        if (targetQCurr == targetQPrev)
          return prevOp;
      }
      for (Value control : quakeOp.getControls()) {
        int controlQPrev =
            extractIndexFromQuakeExtractRefOp(control.getDefiningOp());
        if (targetQCurr == controlQPrev)
          return prevOp;
      }
    }
    // Move to the previous operation
    prevOp = prevOp->getPrevNode();
  }
  return nullptr; // No matching previous operation found
}

// Get the next operation on a given TargeQubit, starting from
// currentOp
Operation *getNextOperationOnTarget(
    Operation *currentOp, Value targetQubit) {
  // Start from the next operation
  Operation *nextOp = currentOp->getNextNode();
  // Iterate through the previous operations in the block
  while (nextOp) {
    // Check if the operation has a target qubit and matches the given target
    if (auto quakeOp = dyn_cast<quake::OperatorInterface>(nextOp)) {
      int targetQCurr =
          extractIndexFromQuakeExtractRefOp(targetQubit.getDefiningOp());
      for (Value target : quakeOp.getTargets()) {
        int targetQNext =
            extractIndexFromQuakeExtractRefOp(target.getDefiningOp());
        if (targetQCurr == targetQNext)
          return nextOp;
      }
      for (Value control : quakeOp.getControls()) {
        int controlQNext =
            extractIndexFromQuakeExtractRefOp(control.getDefiningOp());
        if (targetQCurr == controlQNext)
          return nextOp;
      }
    }
    // Move to the previous operation
    nextOp = nextOp->getNextNode();
  }
  return nullptr; // No matching previous operation found
}
} // namespace mqss::support::quakeDialect