#include "Utils/dataset_conversion.hpp"

// MLIR includes
#include "common/RuntimeMLIR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

// Passes includes
#include "Passes/Transforms.hpp"

// Utils includes
#include "Support/mlir_utils.hpp"
#include "Utils/progress_bar.hpp"

// Stdandard library includes
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string.h>
#include <sys/wait.h>

namespace fs = std::filesystem;
using namespace mqss::opt;
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
int run_shell_command(const std::string &command, const std::string &task) {
  int return_code = 0;
  try {
    return_code = std::system(command.c_str());
  } catch (const std::exception &e) {
    std::cerr << "\nException running " << task << ": " << e.what()
              << std::endl;
    return -1;
  }
  if (return_code != 0) {
    std::cerr << "\n"
              << task << " failed (return code: " << return_code << ")\n";
    return return_code;
  }
  return 0;
}

bool copy_file_and_report(const fs::path &source, const fs::path &destination) {
  std::error_code error_code;
  fs::create_directories(destination.parent_path(), error_code);
  if (!fs::copy_file(source, destination, fs::copy_options::overwrite_existing,
                     error_code)) {
    std::cerr << "\nFailed to copy " << source.string() << " to "
              << destination.string()
              << (error_code ? ": " + error_code.message() : "") << std::endl;
    return false;
  }
  return true;
}

int convert_quake_to_tikz(const fs::path &quake_to_tikz_tool_path,
                          const fs::path &quake_input_path,
                          const fs::path &tikz_output_path,
                          const std::string &append_to_log_file) {
  const std::string command_line = quake_to_tikz_tool_path.string() +
                                   " --input " + quake_input_path.string() +
                                   " --output " + tikz_output_path.string() +
                                   " >> " + append_to_log_file;
  return run_shell_command(command_line, "Conversion qke → tikz for " +
                                             quake_input_path.string());
}

int build_png_from_tikz_file(const fs::path &tikz_file_path) {
  // Minimal wrapper document (standalone) that \input{sometikz.tikz}
  // Write temp.tex next to CWD (consistent with existing workflow).

  const std::string latex_wrapper = "\\documentclass{standalone}\n"
                                    "\\usepackage{tikz}\n"
                                    "\\usepackage{quantikz}\n"
                                    "\\begin{document}\n"
                                    "\\input{" +
                                    tikz_file_path.string() +
                                    "}\n"
                                    "\\end{document}\n";
  std::ofstream temp_tex_file("temp.tex");
  if (!temp_tex_file) {
    std::cerr << "\nFailed to create temp.tex for " << tikz_file_path
              << std::endl;
    return -1;
  }
  temp_tex_file << latex_wrapper;
  temp_tex_file.flush();
  temp_tex_file.close();

  const std::string png_output_base =
      tikz_file_path.string().substr(0, tikz_file_path.string().find(".tikz"));

  // Compile LaTeX and convert to PNG (trimmed). Preserve your flags & quieting.
  const std::string command_line =
      "pdflatex -interaction=nonstopmode temp.tex > /dev/null 2>&1 && "
      "convert -density 300 -strip temp.pdf -trim -quality 90 " +
      png_output_base +
      ".png > /dev/null 2>&1 && "
      "rm temp.* > /dev/null 2>&1";

  return run_shell_command(command_line,
                           "PNG conversion for " + tikz_file_path.string());
}

// ------------------------------------------------------------
// Dataset-wide QASM → Quake conversion (unchanged in behavior)
// ------------------------------------------------------------
void convertAllQasmDatasetsToQuake() {
  try {
    fs::create_directories("./logs");
    const std::string cmd =
        "echo \"qasm_to_quake.log:\n\" > ./logs/qasm_to_quake.log";
    std::system(cmd.c_str());
  } catch (const fs::filesystem_error &e) {
    std::cerr << "\nError creating directory: " << e.what() << std::endl;
    return;
  }
  try {
    if (convertQasmDatasetToQuake("Training") != 0) {
      throw std::runtime_error(
          "Failed to convert Passtest Qasm dataset to Quake.");
    }
    if (convertQasmDatasetToQuake("Passtest") != 0) {
      throw std::runtime_error(
          "Failed to convert Passtest Qasm dataset to Quake.");
    }
    if (convertQasmDatasetToQuake("Tensortest") != 0) {
      throw std::runtime_error(
          "Failed to convert Tensortest Qasm dataset to Quake.");
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }
}

int convertQasmDatasetToQuake(const std::string &subdirectory) {
  const fs::path qasm_dir = fs::path(AI_DATASET_DIR) / "Qasm" / subdirectory;
  const fs::path quake_dir = fs::path(AI_DATASET_DIR) / "Quake" / subdirectory;
  const fs::path tool_path = fs::path(MQSS_BUILD_DIR) / "tools/qasm-to-quake";

  fs::create_directories(quake_dir);
  if (!fs::exists(qasm_dir)) {
    std::cerr << "Input directory does not exist: " << qasm_dir.string()
              << std::endl;
    return -1;
  }

  int total = 0;
  for (const auto &entry : fs::directory_iterator(qasm_dir)) {
    if (!entry.is_regular_file()) {
      std::cout << "Rejecting conversion qasm->quake for: " << entry
                << " because it's not a regular file." << std::endl;
      continue;
    }
    if (entry.path().extension() != ".qasm") {
      std::cout << "Rejecting conversion qasm->quake for: " << entry
                << " because extension is not .qasm." << std::endl;
      continue;
    }
    total++;
  }
  if (total == 0) {
    std::cout << "No .qasm files found for qasm to quake conversion."
              << std::endl;
    return -1;
  }
  std::cout << "Converting " << total << " Qasm circuits to Quake from "
            << quake_dir.string() << " dataset." << std::endl;

  int current = 0;
  for (const auto &entry : fs::directory_iterator(qasm_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".qasm") {
      fs::path input = entry.path();
      std::string circuit_name = input.stem().string();
      fs::path output = quake_dir / (circuit_name + ".qke");
      updateProgress(current, total, circuit_name);
      std::string cmd = tool_path.string() + " --input " + input.string() +
                        " --output " + output.string() +
                        " >> ./logs/qasm_to_quake.log";
      int ret = 0;
      try {
        ret = std::system(cmd.c_str());
      } catch (const std::exception &e) {
        std::cerr << "\nException on " << input << ": " << e.what()
                  << std::endl;
      }
      if (ret != 0) {
        std::cerr << "\nConversion failed for " << input
                  << " (return code: " << ret << ")" << std::endl;
        return -1;
      }

      updateProgress(++current, total, "");
    }
  }
  std::cout << std::endl;
  return 0;
}
} // namespace ai_pass_selector