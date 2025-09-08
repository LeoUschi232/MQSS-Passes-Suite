#include "mlir_utils.hpp"
#include "Torch/environment.hpp"

#include <torch/torch.h>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
  const fs::path filename = fs::path(AI_DATASET_DIR) /
                            "Quake/Passtest/CrxToHCrzH_input.qke";

  std::string quakeModule = getQuake(filename.string());
  auto [mlirModule, _] = extractMLIRContext(quakeModule);
  auto env = ai_pass_selector::QuantumCircuitEnviorment(4, 20, 10, mlirModule);
  env.get_circuit_info();
  return 0;
}