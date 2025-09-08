#ifndef MQSS_PASSES_SUITE_ENVIRONMENT_HPP
#define MQSS_PASSES_SUITE_ENVIRONMENT_HPP

//  Quake includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"

// MLIR includes
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

// Passes includes
#include "Passes/Cancellations.hpp"
#include "Passes/CodeGen.hpp"
#include "Passes/Decompositions.hpp"
#include "Passes/Examples.hpp"
#include "Passes/Transforms.hpp"

// AI Utils includes
#include "Utils/mlir_utils.hpp"
#include "Utils/progress_bar.hpp"

// Stdandard library includes
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/wait.h>

namespace ai_pass_selector {
int printOperation(Operation * op);

void f1(const std::string &subdirectory);


} // namespace ai_pass_selector

#endif // MQSS_PASSES_SUITE_ENVIRONMENT_HPP