#include "Utils/conversion_workflow.hpp"

// MLIR includes
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

// Passes includes
#include "Passes/Cancellations.hpp"
#include "Passes/CodeGen.hpp"
#include "Passes/Decompositions.hpp"
#include "Passes/Transforms.hpp"

// AI Utils includes
#include "Utils/mlir_utils.hpp"
#include "Utils/progress_bar.hpp"

// Stdandard library includes
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <string.h>
#include <sys/wait.h>

namespace fs = std::filesystem;
using namespace mqss::opt;

namespace ai_pass_selector {
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
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }
}

int convertQasmDatasetToQuake(const std::string &subdirectory) {
  const fs::path qasm_dir = fs::path(AI_DATASET_DIR) / "Qasm" / subdirectory;
  const fs::path quake_dir = fs::path(AI_DATASET_DIR) / "Quake" / subdirectory;
  const fs::path tool_path = fs::path(MQSS_BUILD_DIR) / "tools/qasm-to-quake";

  // Create output dir if it doesn't exist.
  fs::create_directories(quake_dir);
  if (!fs::exists(qasm_dir)) {
    std::cerr << "Input directory does not exist: " << qasm_dir.string()
        << std::endl;
    return -1;
  }

  // First pass: count the number of .qasm files.
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

  // Second pass: process files and update progress.
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
      if (const int ret = std::system(cmd.c_str()); ret != 0) {
        std::cerr << "\nConversion failed for " << input
            << " (return code: " << ret << ")" << std::endl;
        return -1;
      }

      updateProgress(++current, total, "");
    }
  }
  // Move to next line after progress bar completes.
  std::cout << std::endl;
  return 0;
}

void convertAllPasstestCircuitsToTikz() {
  std::vector<
        std::tuple<std::string, std::function<std::unique_ptr<mlir::Pass>()> >
      >
      passes = {
          {"ZeroRxToId", [] { return createZeroRxToIdPass(); }},
          {"ZeroRyToId", [] { return createZeroRyToIdPass(); }},
          {"ZeroRzToId", [] { return createZeroRzToIdPass(); }},
          {"CxCxToId", [] { return createCxCxToIdPass(); }},
          {"CyCyToId", [] { return createCyCyToIdPass(); }},
          {"CzCzToId", [] { return createCzCzToIdPass(); }},
          {"XXToId", [] { return createXXToIdPass(); }},
          {"YYToId", [] { return createYYToIdPass(); }},
          {"ZZToId", [] { return createZZToIdPass(); }},
          {"SSdgToId", [] { return createSSdgToIdPass(); }},
          {"SdgSToId", [] { return createSdgSToIdPass(); }},
          {"TTdgToId", [] { return createTTdgToIdPass(); }},
          {"TdgTToId", [] { return createTdgTToIdPass(); }},
          {"HHToId", [] { return createHHToIdPass(); }},
          {"RxRxToRx", [] { return createRxRxToRxPass(); }},
          {"RyRyToRy", [] { return createRyRyToRyPass(); }},
          {"RzRzToRz", [] { return createRzRzToRzPass(); }},
          {"HXHToZ", [] { return createHXHToZPass(); }},
          {"HZHToX", [] { return createHZHToXPass(); }},
          {"XHZToH", [] { return createXHZToHPass(); }},
          {"ZHXToH", [] { return createZHXToHPass(); }},
          {"HRxHToRz", [] { return createHRxHToRzPass(); }},
          {"HRzHToRx", [] { return createHRzHToRxPass(); }},
          {"HCxHToCz", [] { return createHCxHToCzPass(); }},
          {"HCzHToCx", [] { return createHCzHToCxPass(); }},
          {"HCrxHToCrz", [] { return createHCrxHToCrzPass(); }},
          {"HCrzHToCrx", [] { return createHCrzHToCrxPass(); }},
          {"XHToHZ", [] { return createXHToHZPass(); }},
          {"HXToZH", [] { return createHXToZHPass(); }},
          {"YHToHY", [] { return createYHToHYPass(); }},
          {"HYToYH", [] { return createHYToYHPass(); }},
          {"ZHToHX", [] { return createZHToHXPass(); }},
          {"HZToXH", [] { return createHZToXHPass(); }},
          {"SSSToSdg", [] { return createSSSToSdgPass(); }},
          {"SdgSdgSdgToS", [] { return createSdgSdgSdgToSPass(); }},
          {"SSToZ", [] { return createSSToZPass(); }},
          {"SdgSdgToZ", [] { return createSdgSdgToZPass(); }},
          {"SZToSdg", [] { return createSZToSdgPass(); }},
          {"ZSToSdg", [] { return createZSToSdgPass(); }},
          {"SdgZToS", [] { return createSdgZToSPass(); }},
          {"ZSdgToS", [] { return createZSdgToSPass(); }},
          {"TTToS", [] { return createTTToSPass(); }},
          {"CxCxCxToSwap", [] { return createCxCxCxToSwapPass(); }},
          {"CxZToZCx", [] { return createCxZToZCxPass(); }},
          {"ZCxToCxZ", [] { return createZCxToCxZPass(); }},
          {"CxXToXCx", [] { return createCxXToXCxPass(); }},
          {"XCxToCxX", [] { return createXCxToCxXPass(); }},
          {"CxRxToRxCx", [] { return createCxRxToRxCxPass(); }},
          {"RxCxToCxRx", [] { return createRxCxToCxRxPass(); }},
          {"ReverseCx", [] { return createReverseCxPass(); }},
          {"XToHZH", [] { return createXToHZHPass(); }},
          {"ZToHXH", [] { return createZToHXHPass(); }},
          {"RxToHRzH", [] { return createRxToHRzHPass(); }},
          {"RzToHRxH", [] { return createRzToHRxHPass(); }},
          {"CxToUpperHCzH", [] { return createCxToUpperHCzHPass(); }},
          {"CxToLowerHCzH", [] { return createCxToLowerHCzHPass(); }},
          {"CzToUpperHCxH", [] { return createCzToUpperHCxHPass(); }},
          {"CzToLowerHCxH", [] { return createCzToLowerHCxHPass(); }},
          {"CrxToHCrzH", [] { return createCrxToHCrzHPass(); }},
          {"CrzToHCrxH", [] { return createCrzToHCrxHPass(); }},
          {"SdgToSSS", [] { return createSdgToSSSPass(); }},
          {"SToSdgSdgSdg", [] { return createSToSdgSdgSdgPass(); }},
          {"SToTT", [] { return createSToTTPass(); }},
          {"SwapToLowerCxCxCx", [] { return createSwapToLowerCxCxCxPass(); }},
          {"SwapToUpperCxCxCx",
           [] { return createSwapToUpperCxCxCxPass(); }}};
  int total = passes.size();
  std::cout << "Converting " << total << " quake passtest circuits to tikz."
      << std::endl;

  if (total == 0) {
    std::cout << "No passes found for conversion to TikZ." << std::endl;
    return;
  }

  try {
    fs::create_directories("./logs");
    std::string cmd =
        "echo \"passtest_to_tikz_before.log:\n\" > ./logs/passtest_to_tikz_before.log";
    std::system(cmd.c_str());
    cmd =
        "echo \"passtest_to_tikz_after.log:\n\" > ./logs/passtest_to_tikz_after.log";
    std::system(cmd.c_str());
  } catch (const fs::filesystem_error &e) {
    std::cerr << "\nError creating directory: " << e.what() << std::endl;
    return;
  }
  int current = 0;
  for (const auto &[passname, creator] : passes) {
    updateProgress(current, total, passname);
    const pid_t pid = fork();
    if (pid == 0) {
      exit(convertPasstestCircuitToTikz(passname, creator()));
    }
    int status;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) {
      std::cerr << "Pass: " << passname << " crashed with signal "
          << strsignal(WTERMSIG(status)) << std::endl;
      break;
    }
    updateProgress(++current, total, "");
  }
  // Move to next line after progress bar completes.
  std::cout << std::endl;
  const fs::path latex_dir = fs::path(AI_DATASET_DIR) / "Latex";
  const fs::path tex_file = latex_dir / "passtest_quantum_circuits.tex";
  const std::string cmd =
      "pdflatex --shell-escape -output-directory=" + latex_dir.string() + " " +
      tex_file.string() + " > /dev/null 2>&1";
  std::cout << "Compiling LaTeX file to PDF." << std::endl;
  if (const int ret = std::system(cmd.c_str()); ret != 0) {
    std::cerr << "\nPDF generation failed for passtest_quantum_circuits.tex "
        << "(return code: " << ret << ")" << std::endl;
    return;
  }
  std::cout << "PDF generated successfully at "
      << (latex_dir / "passtest_quantum_circuits.pdf").string()
      << std::endl;
}

int convertPasstestCircuitToTikz(std::string passname,
                                 std::unique_ptr<mlir::Pass> pass) {
  fs::path quake_src =
      fs::path(AI_DATASET_DIR) / "Quake/Passtest" / (passname + "_input.qke");
  fs::path latex_qke_input =
      fs::path(AI_DATASET_DIR) / "Latex/Passtest" / (passname + "_input.qke");
  fs::path latex_qke_output =
      fs::path(AI_DATASET_DIR) / "Latex/Passtest" / (passname + "_output.qke");
  fs::path latex_tikz_input =
      fs::path(AI_DATASET_DIR) / "Latex/Passtest" / (passname + "_input.tikz");
  fs::path latex_tikz_output =
      fs::path(AI_DATASET_DIR) / "Latex/Passtest" / (passname + "_output.tikz");
  fs::path quake_to_tikz_path =
      fs::path(MQSS_BUILD_DIR) / "tools/quake-to-tikz";

  if (!fs::copy_file(quake_src, latex_qke_input,
                     fs::copy_options::overwrite_existing)) {
    std::cerr << "\nFailed to copy " << quake_src.string() << " to "
        << latex_qke_input.string() << std::endl;
    return -1;
  }
  std::string cmd = quake_to_tikz_path.string() + " --input " +
                    latex_qke_input.string() + " --output " +
                    latex_tikz_input.string() +
                    " >> ./logs/passtest_to_tikz_before.log";
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "\nConversion failed for " << latex_qke_input
        << " (return code: " << ret << ")" << std::endl;
    return -1;
  }

  std::string quakeModule = getQuake(quake_src.string());
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  mlir::PassManager pm(&context);
  pm.addPass(std::move(pass));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (mlir::failed(pm.run(mlirModule))) {
    std::cerr << "\nThe pass failed for " << quake_src.string() << std::endl;
    return -1;
  }

  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  mlirModule->print(stringStream);
  std::ofstream outputFile(latex_qke_output.string());
  if (!outputFile) {
    std::cerr << "\nFailed to open " << latex_qke_output.string() << std::endl;
    return -1;
  }
  outputFile << moduleOutput;
  outputFile.close();

  cmd = quake_to_tikz_path.string() + " --input " + latex_qke_output.string() +
        " --output " + latex_tikz_output.string() +
        " >> ./logs/passtest_to_tikz_after.log";
  ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "\nConversion failed for " << latex_qke_output
        << " (return code: " << ret << ")" << std::endl;
    return -1;
  }

  // Create temp.tex for input
  std::string temp_tex_content = "\\documentclass{standalone}\n"
                                 "\\usepackage{tikz}\n"
                                 "\\usetikzlibrary{quantikz}\n"
                                 "\\begin{document}\n"
                                 "\\input{" +
                                 latex_tikz_input.string() +
                                 "}\n"
                                 "\\end{document}\n";
  std::ofstream tempFile("temp.tex");
  if (!tempFile) {
    std::cerr << "\nFailed to create temp.tex for " << latex_tikz_input
        << std::endl;
    return -1;
  }
  tempFile << temp_tex_content;
  tempFile.close();

  // Compile and convert input
  cmd = "pdflatex -interaction=nonstopmode temp.tex > /dev/null 2>&1 && "
        "convert -density 300 -strip temp.pdf -trim -quality 90 " +
        latex_tikz_input.string().substr(
            0, latex_tikz_input.string().find(".tikz")) +
        ".png > /dev/null 2>&1 && "
        "rm temp.* > /dev/null 2>&1";
  ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "\nPNG conversion failed for " << latex_tikz_input
        << " (return code: " << ret << ")" << std::endl;
    return -1;
  }

  // Create temp.tex for output
  temp_tex_content = "\\documentclass{standalone}\n"
                     "\\usepackage{tikz}\n"
                     "\\usetikzlibrary{quantikz}\n"
                     "\\begin{document}\n"
                     "\\input{" +
                     latex_tikz_output.string() +
                     "}\n"
                     "\\end{document}\n";
  tempFile.open("temp.tex");
  if (!tempFile) {
    std::cerr << "\nFailed to create temp.tex for " << latex_tikz_output
        << std::endl;
    return -1;
  }
  tempFile << temp_tex_content;
  tempFile.close();

  // Compile and convert output
  cmd = "pdflatex -interaction=nonstopmode temp.tex > /dev/null 2>&1 && "
        "convert -density 300 -strip temp.pdf -trim -quality 90 " +
        latex_tikz_output.string().substr(
            0, latex_tikz_output.string().find(".tikz")) +
        ".png > /dev/null 2>&1 && "
        "rm temp.* > /dev/null 2>&1";
  ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "\nPNG conversion failed for " << latex_tikz_output
        << " (return code: " << ret << ")" << std::endl;
    return -1;
  }
  return 0;
}
} // namespace ai_pass_selector