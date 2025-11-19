/* This code and any associated documentation is provided "as is"

Copyright 2024 Munich Quantum Software Stack Project

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
  date   December 2024
  version 1.0
  brief
  This file contains the unitary tests for each MLIR pass in the Munich Quantum
  Software Stack (MQSS).
  * Folder code has quantum kernels written in CudaQ (cpp).
  * Folder golden contains the expected modified quantum kernel in MLIR for each
    MLIR pass.
  1. In each test, the quantum kernel in CudaQ is lowered to QUAKE MLIR.
  2. Then the pass is applied to the QUAKE MLIR kernel.
  3. The output of the pass is compared to the expected output.
  4. Success if both expected output and the output obtained by the pass
matches.

******************************************************************************/

#include <iostream>
#include <string>
// llvm includes
#include "llvm/Support/raw_ostream.h"
// mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h" // For translateModuleToLLVMIR
#include "mlir/Transforms/Passes.h"
// cudaq includes
// includes in runtime
#include "common/RuntimeMLIR.h"
// includes mqss passes
#include "Passes/Cancellations.hpp"
#include "Passes/CodeGen.hpp"
#include "Passes/Decompositions.hpp"
#include "Passes/Examples.hpp"
#include "Passes/Transforms.hpp"
// test includes
#include "mlir_utils.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <sc/Architecture.hpp>
#include <sc/configuration/Configuration.hpp>
#include <sc/configuration/Heuristic.hpp>
#include <sc/configuration/InitialLayout.hpp>
#include <sc/configuration/Layering.hpp>
#include <sc/configuration/LookaheadHeuristic.hpp>
#include <sc/utils.hpp>

#define CUDAQ_GEN_PREFIX_NAME "__nvqpp__mlirgen__"

using namespace mqss::opt;
using namespace mqss::support::quakeDialect;

std::tuple<ModuleOp, MLIRContext *> createEmptyMLIRModule() {
  auto contextPtr = cudaq::initializeMLIR();
  MLIRContext &context = *contextPtr.get();
  // Create an empty MLIR module
  mlir::OwningOpRef m_module =
      ModuleOp::create(mlir::UnknownLoc::get(&context));
  return std::make_tuple(m_module.release(), contextPtr.release());
}

std::tuple<std::string, std::string>
getQuakeAndGolden(const std::string &inputFile, const std::string &goldenFile) {
  std::string quakeModule = readFileToString(inputFile);
  std::string goldenOutput = readFileToString(goldenFile);
  return std::make_tuple(quakeModule, goldenOutput);
}

std::string normalize(const std::string &str) {
  std::string result;
  for (const char c : str) {
    if (c != '\t' && c != '\n' && c != '\\' && c != ' ') {
      result += c;
    }
  }
  return result;
}

TEST(TestMQSSPasses, TestPrintQuakeGatesPass) {
  // load mlir module and the golden output
  auto [quakeModule, goldenOutput] =
      getQuakeAndGolden("./quake/PrintQuakeGatesPass.qke",
                        "./golden-cases/PrintQuakeGatesPass.qke");
#ifdef DEBUG
  std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
#endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Adding custom pass
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  pm.addPass(createPrintQuakeGatesPass(stringStream));
  // running the pass
  if (mlir::failed(pm.run(mlirModule)))
    throw std::runtime_error("The pass failed...");
#ifdef DEBUG
  std::cout << "Captured output from Pass:\n" << moduleOutput << std::endl;
#endif
  EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

TEST(TestMQSSPasses, TestQuakeQMapPass01) {
  // load mlir module and the golden output
  auto [quakeModule, goldenOutput] = getQuakeAndGolden(
      "./quake/QuakeQMapPass-01.qke", "./golden-cases/QuakeQMapPass-01.qke");
#ifdef DEBUG
  std::cout << "Input Quake Module 01 " << std::endl
            << quakeModule << std::endl;
#endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Defining test architecture
  Architecture arch{};
  /*
      3
     / \
    4   2
    |   |
    0---1
  */
  const CouplingMap cm = {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3},
                          {3, 2}, {3, 4}, {4, 3}, {4, 0}, {0, 4}};
  arch.loadCouplingMap(5, cm);
#ifdef DEBUG
  std::cout << "Dumping the architecture " << std::endl;
  Architecture::printCouplingMap(arch.getCouplingMap(), std::cout);
#endif
  // Defining the settings of the mqt-mapper
  Configuration settings{};
  settings.heuristic = Heuristic::GateCountMaxDistance;
  settings.layering = Layering::DisjointQubits;
  settings.initialLayout = InitialLayout::Identity;
  settings.preMappingOptimizations = false;
  settings.postMappingOptimizations = false;
  settings.lookaheadHeuristic = LookaheadHeuristic::None;
  settings.debug = false;
  settings.addMeasurementsToMappedCircuit = true;
  // Adding the QuakeQMap pass to the PassManager
  pm.nest<FuncOp>().addPass(createQuakeQMapPass(arch, settings));
  // pass to canonical form and remove non-used operations
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // running the pass
  if (mlir::failed(pm.run(mlirModule))) {
    throw std::runtime_error("The pass failed...");
  }
#ifdef DEBUG
  std::cout << "Mapped Circuit:\n";
  mlirModule->dump();
#endif
  // Convert the module to a string
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  mlirModule->print(stringStream);
  EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

TEST(TestMQSSPasses, TestQuakeQMapPass02) {
  // load mlir module and the golden output
  auto [quakeModule, goldenOutput] = getQuakeAndGolden(
      "./quake/QuakeQMapPass-02.qke", "./golden-cases/QuakeQMapPass-02.qke");
#ifdef DEBUG
  std::cout << "Input Quake Module 01 " << std::endl
            << quakeModule << std::endl;
#endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Defining test architecture
  Architecture arch{};
  /*
      3
     / \
    4   2
    |   |
    0---1
  */
  const CouplingMap cm = {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3},
                          {3, 2}, {3, 4}, {4, 3}, {4, 0}, {0, 4}};
  arch.loadCouplingMap(5, cm);
#ifdef DEBUG
  std::cout << "Dumping the architecture " << std::endl;
  Architecture::printCouplingMap(arch.getCouplingMap(), std::cout);
#endif
  // Defining the settings of the mqt-mapper
  Configuration settings{};
  settings.heuristic = Heuristic::GateCountMaxDistance;
  settings.layering = Layering::DisjointQubits;
  settings.initialLayout = InitialLayout::Identity;
  settings.preMappingOptimizations = false;
  settings.postMappingOptimizations = false;
  settings.lookaheadHeuristic = LookaheadHeuristic::None;
  settings.debug = false;
  settings.addMeasurementsToMappedCircuit = true;
  // Adding the QuakeQMap pass to the PassManager
  pm.nest<mlir::func::FuncOp>().addPass(
      mqss::opt::createQuakeQMapPass(arch, settings));
  // pass to canonical form and remove non-used operations
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // running the pass
  if (mlir::failed(pm.run(mlirModule))) {
    throw std::runtime_error("The pass failed...");
  }
#ifdef DEBUG
  std::cout << "Mapped Circuit:\n";
  mlirModule->dump();
#endif
  // Convert the module to a string
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  mlirModule->print(stringStream);
  EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

TEST(TestMQSSPasses, TestQuakeToTikzPass) {
  // load mlir module and the golden output
  auto [quakeModule, goldenOutput] = getQuakeAndGolden(
      "./quake/QuakeToTikzPass.qke", "./golden-cases/QuakeToTikzPass.tikz.tex");
#ifdef DEBUG
  std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
#endif

  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Adding custom pass
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  pm.nest<mlir::func::FuncOp>().addPass(
      mqss::opt::createQuakeToTikzPass(stringStream));
  // running the pass
  if (mlir::failed(pm.run(mlirModule))) {
    throw std::runtime_error("The pass failed...");
  }
#ifdef DEBUG
  std::cout << "Captured output from Pass:\n" << moduleOutput << std::endl;
#endif
  EXPECT_EQ(normalize(goldenOutput), normalize(moduleOutput));
}

std::tuple<std::string, std::string>
behaviouralTest(std::tuple<std::string, std::string, std::string,
                           std::function<std::unique_ptr<mlir::Pass>()>, bool>
                    test) {
  std::string fileInputTest = std::get<1>(test);
  std::string fileGoldenCase = std::get<2>(test);
  auto passMlir = std::get<3>(test);
  bool embeddedPass = std::get<4>(test);
  // Invoke the function to create the pass
  std::unique_ptr<mlir::Pass> pass = passMlir();

  // load mlir module and the golden output
  auto [quakeModule, goldenOutput] =
      getQuakeAndGolden(fileInputTest, fileGoldenCase);
#ifdef DEBUG
  std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
#endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Adding the pass to the PassManager
  if (embeddedPass) {
    pm.nest<mlir::func::FuncOp>().addPass(std::move(pass));
  } else
    pm.addPass(std::move(pass));
  // pm.addPass(std::move(pass));
  // pass to canonical form and remove non-used operations
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // running the pass
  if (mlir::failed(pm.run(mlirModule))) {
    throw std::runtime_error("The pass failed...");
  }
#ifdef DEBUG
  std::cout << "Circuit after pass:\n";
  mlirModule->dump();
#endif
  // Convert the module to a string
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  mlirModule->print(stringStream);
  return std::make_tuple(goldenOutput, moduleOutput);
}

class BehaviouralTestPassesMQSS
    : public ::testing::TestWithParam<
          std::tuple<std::string, std::string, std::string,
                     std::function<std::unique_ptr<mlir::Pass>()>, bool>> {};

TEST_P(BehaviouralTestPassesMQSS, Run) {
  const std::tuple<std::string, std::string, std::string,
                   std::function<std::unique_ptr<mlir::Pass>()>, bool>
      p = GetParam();
  const std::string testName = std::get<0>(p);
  SCOPED_TRACE(testName);
  auto [goldenOutput, moduleOutput] = behaviouralTest(p);
  EXPECT_EQ(goldenOutput, std::string(moduleOutput));
}

INSTANTIATE_TEST_SUITE_P(
    MQSSPassTests, BehaviouralTestPassesMQSS,
    ::testing::Values(
        std::make_tuple(
            "TestCustomExamplePass", "./quake/CustomExamplePass.qke",
            "./golden-cases/CustomExamplePass.qke",
            []() { return mqss::opt::createCustomExamplePass(); }, false),
        std::make_tuple(
            "TestCxToUpperHCzHPass", "./quake/CxToHCzHPass.qke",
            "./golden-cases/CxToHCzHPass.qke",
            []() { return mqss::opt::createCxToUpperHCzHPass(); }, false),
        std::make_tuple(
            "TestCxToLowerHCzHPass", "./quake/CxToHCzHPass.qke",
            "./golden-cases/CxToHCzHPass.qke",
            []() { return mqss::opt::createCxToLowerHCzHPass(); }, false),
        std::make_tuple(
            "TestCzToUpperHCxHPass", "./quake/CzToHCxHPass.qke",
            "./golden-cases/CzToHCxHPass.qke",
            []() { return mqss::opt::createCzToUpperHCxHPass(); }, false),
        std::make_tuple(
            "TestCzToLowerHCxHPass", "./quake/CzToHCxHPass.qke",
            "./golden-cases/CzToHCxHPass.qke",
            []() { return mqss::opt::createCzToLowerHCxHPass(); }, false),
        std::make_tuple(
            "TestCommuteCnotRxPass", "./quake/CommuteCNotRxPass.qke",
            "./golden-cases/CommuteCNotRxPass.qke",
            []() { return mqss::opt::createCxRxToRxCxPass(); }, false),
        std::make_tuple(
            "TestCommuteCnotXPass", "./quake/CommuteCNotXPass.qke",
            "./golden-cases/CommuteCNotXPass.qke",
            []() { return mqss::opt::createCxXToXCxPass(); }, false),
        std::make_tuple(
            "TestCommuteCnotZPass01", "./quake/CommuteCNotZPass-01.qke",
            "./golden-cases/CommuteCNotZPass-01.qke",
            []() { return mqss::opt::createCxZToZCxPass(); }, false),
        std::make_tuple(
            "TestCommuteCnotZPass", "./quake/CommuteCNotZPass.qke",
            "./golden-cases/CommuteCNotZPass.qke",
            []() { return mqss::opt::createCxZToZCxPass(); }, false),
        std::make_tuple(
            "TestCommuteRxCnotPass", "./quake/CommuteRxCNotPass.qke",
            "./golden-cases/CommuteRxCNotPass.qke",
            []() { return mqss::opt::createRxCxToCxRxPass(); }, false),
        std::make_tuple(
            "TestCommuteXCNotPass", "./quake/CommuteXCNotPass.qke",
            "./golden-cases/CommuteXCNotPass.qke",
            []() { return mqss::opt::createXCxToCxXPass(); }, false),
        std::make_tuple(
            "TestCommuteZCnotPass", "./quake/CommuteZCNotPass.qke",
            "./golden-cases/CommuteZCNotPass.qke",
            []() { return mqss::opt::createZCxToCxZPass(); }, false),
        std::make_tuple(
            "TestCommuteZCnotPass01", "./quake/CommuteZCNotPass-01.qke",
            "./golden-cases/CommuteZCNotPass-01.qke",
            []() { return mqss::opt::createZCxToCxZPass(); }, false),
        std::make_tuple(
            "CxCxToIdPass", "./quake/CxCxToIdPass.qke",
            "./golden-cases/CxCxToIdPass.qke",
            []() { return mqss::opt::createCxCxToIdPass(); }, false),
        std::make_tuple(
            "ReverseCNotPass", "./quake/ReverseCNotPass.qke",
            "./golden-cases/ReverseCNotPass.qke",
            []() { return mqss::opt::createReverseCxPass(); }, false),
        std::make_tuple(
            "HXHToZPass", "./quake/HXHToZPass.qke",
            "./golden-cases/HXHToZPass.qke",
            []() { return mqss::opt::createHXHToZPass(); }, false),
        std::make_tuple(
            "XGateAndHadamardSwitchPass",
            "./quake/XGateAndHadamardSwitchPass.qke",
            "./golden-cases/XGateAndHadamardSwitchPass.qke",
            []() { return mqss::opt::createXHToHZPass(); }, false),
        std::make_tuple(
            "YGateAndHadamardSwitchPass",
            "./quake/YGateAndHadamardSwitchPass.qke",
            "./golden-cases/YGateAndHadamardSwitchPass.qke",
            []() { return mqss::opt::createYHToHYPass(); }, false),
        std::make_tuple(
            "ZGateAndHadamardSwitchPass",
            "./quake/ZGateAndHadamardSwitchPass.qke",
            "./golden-cases/ZGateAndHadamardSwitchPass.qke",
            []() { return mqss::opt::createZHToHXPass(); }, false),
        std::make_tuple(
            "HZHToXPass", "./quake/HZHToXPass.qke",
            "./golden-cases/HZHToXPass.qke",
            []() { return mqss::opt::createHZHToXPass(); }, false),
        std::make_tuple(
            "HadamardAndXGateSwitchPass",
            "./quake/HadamardAndXGateSwitchPass.qke",
            "./golden-cases/HadamardAndXGateSwitchPass.qke",
            []() { return mqss::opt::createHXToZHPass(); }, false),
        std::make_tuple(
            "HadamardAndYGateSwitchPass",
            "./quake/HadamardAndYGateSwitchPass.qke",
            "./golden-cases/HadamardAndYGateSwitchPass.qke",
            []() { return mqss::opt::createHYToYHPass(); }, false),
        std::make_tuple(
            "HadamardAndZGateSwitchPass",
            "./quake/HadamardAndZGateSwitchPass.qke",
            "./golden-cases/HadamardAndZGateSwitchPass.qke",
            []() { return mqss::opt::createHZToXHPass(); }, false),
        std::make_tuple(
            "ZeroRxToIdPass", "./quake/ZeroRxToIdPass.qke",
            "./golden-cases/ZeroRxToIdPass.qke",
            []() { return mqss::opt::createZeroRxToIdPass(); }, false),
        std::make_tuple(
            "ZeroRyToIdPass", "./quake/ZeroRyToIdPass.qke",
            "./golden-cases/ZeroRyToIdPass.qke",
            []() { return mqss::opt::createZeroRyToIdPass(); }, false),
        std::make_tuple(
            "ZeroRzToIdPass", "./quake/ZeroRzToIdPass.qke",
            "./golden-cases/ZeroRzToIdPass.qke",
            []() { return mqss::opt::createZeroRzToIdPass(); }, false),
        std::make_tuple(
            "XXToIdPass", "./quake/XXToIdPass.qke",
            "./golden-cases/XXToIdPass.qke",
            []() { return mqss::opt::createXXToIdPass(); }, false),
        std::make_tuple(
            "YYToIdPass", "./quake/YYToIdPass.qke",
            "./golden-cases/YYToIdPass.qke",
            []() { return mqss::opt::createYYToIdPass(); }, false),
        std::make_tuple(
            "ZZToIdPass", "./quake/ZZToIdPass.qke",
            "./golden-cases/ZZToIdPass.qke",
            []() { return mqss::opt::createZZToIdPass(); }, false),
        std::make_tuple(
            "SdgZToSPass", "./quake/SAdjToSPass.qke",
            "./golden-cases/SAdjToSPass.qke",
            []() { return mqss::opt::createSdgZToSPass(); }, false),
        std::make_tuple(
            "SZToSdgPass", "./quake/SToSAdjPass.qke",
            "./golden-cases/SToSAdjPass.qke",
            []() { return mqss::opt::createSZToSdgPass(); }, false),
        std::make_tuple(
            "NormalizeArgAnglePass", "./quake/NormalizeArgAnglePass.qke",
            "./golden-cases/NormalizeArgAnglePass.qke",
            []() { return mqss::opt::createNormalizeArgAnglePass(); }, false)),
    [](const ::testing::TestParamInfo<BehaviouralTestPassesMQSS::ParamType>
           &info) {
      // Use the first element of the tuple (testName) as the custom test name
      return std::get<0>(info.param);
    });

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}