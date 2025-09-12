#include "mlir_utils.hpp"
#include "Environment/environment.hpp"

#include <torch/torch.h>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <relative-path-in-dataset>\n";
    return 1;
  }

  const fs::path filename = fs::path(AI_DATASET_DIR) / argv[1];

  std::string quakeModule = getQuake(filename.string());
  auto [mlirModule, _] = extractMLIRContext(quakeModule);
  auto env = ai_pass_selector::QuantumCircuitEnviorment(4, 20, 10, mlirModule);
  auto circuit_info = env.get_circuit_info();

  std::cout << "Circuit info:\n";
  for (const auto &[key, value] : circuit_info)
    std::cout << "  " << key << ": " << value << "\n";

  return 0;
}