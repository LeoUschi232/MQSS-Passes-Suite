// Include the header relative to the include directory.
#include "Utils/dataset_conversion.hpp"

#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  if (argc < 2) {
    ai_pass_selector::convertAllQasmDatasetsToQuake();
  } else {
    ai_pass_selector::convertQasmDatasetToQuake(std::string(argv[1]));
  }
  return 0;
}