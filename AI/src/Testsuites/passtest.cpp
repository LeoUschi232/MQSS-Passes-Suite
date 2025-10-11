#include "Testsuites/passtest.hpp"

// MLIR includes
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

// Passes includes
#include "Passes/Cancellations.hpp"
#include "Passes/Decompositions.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"

// Utils includes
#include "Utils/dataset_conversion.hpp"
#include "Utils/progress_bar.hpp"

// Stdandard library includes
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string.h>
#include <sys/wait.h>
#include <vector>

namespace fs = std::filesystem;
using namespace mqss::opt;
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
// ------------------------------------------------------------
// Passtest → TikZ PNGs
// ------------------------------------------------------------
void convertAllPasstestCircuitsToTikz() {
  std::vector<
      std::tuple<std::string, std::function<std::unique_ptr<mlir::Pass>()>>>
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
          {"SwapToUpperCxCxCx", [] { return createSwapToUpperCxCxCxPass(); }}};

  const int total = static_cast<int>(passes.size());
  std::cout << "Converting " << total << " quake passtest circuits to tikz."
            << std::endl;

  if (total == 0) {
    std::cout << "No passes found for conversion to TikZ." << std::endl;
    return;
  }

  try {
    fs::create_directories("./logs");
    std::string cmd = "echo \"passtest_to_tikz_before.log:\n\" > "
                      "./logs/passtest_to_tikz_before.log";
    std::system(cmd.c_str());
    cmd = "echo \"passtest_to_tikz_after.log:\n\" > "
          "./logs/passtest_to_tikz_after.log";
    std::system(cmd.c_str());
  } catch (const fs::filesystem_error &e) {
    std::cerr << "\nError creating directory: " << e.what() << std::endl;
    return;
  }

  int current = 0;
  for (const auto &[passname, pass_creator] : passes) {
    updateProgress(current, total, passname);
    const pid_t child_pid = fork();
    if (child_pid == 0) {
      exit(convertPasstestCircuitToTikz(passname, pass_creator()));
    }
    int status;
    waitpid(child_pid, &status, 0);
    if (WIFSIGNALED(status)) {
      std::cerr << "\nPass: " << passname << " crashed with signal "
                << strsignal(WTERMSIG(status)) << std::endl;
      break;
    }
    updateProgress(++current, total, "");
  }
  std::cout << std::endl;

  const fs::path latex_dir = fs::path(AI_DATASET_DIR) / "Latex";
  const fs::path tex_file = latex_dir / "passtest_quantum_circuits.tex";
  const std::string compile_pdf =
      "pdflatex -interaction=nonstopmode -halt-on-error --shell-escape "
      "-output-directory=" +
      latex_dir.string() + " " + tex_file.string() + " > /dev/null 2>&1";
  std::cout << "\nCompiling LaTeX file to PDF." << std::endl;
  if (const int ret = std::system(compile_pdf.c_str()); ret != 0) {
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
  const fs::path quake_source_input_file =
      fs::path(AI_DATASET_DIR) / "Quake/Passtest" / (passname + "_input.qke");

  const fs::path latex_passtest_dir =
      fs::path(AI_DATASET_DIR) / "Latex/Passtest";
  const fs::path latex_quake_input_file =
      latex_passtest_dir / (passname + "_input.qke");
  const fs::path latex_quake_output_file =
      latex_passtest_dir / (passname + "_output.qke");
  const fs::path latex_tikz_input_file =
      latex_passtest_dir / (passname + "_input.tikz");
  const fs::path latex_tikz_output_file =
      latex_passtest_dir / (passname + "_output.tikz");

  const fs::path quake_to_tikz_tool_path =
      fs::path(MQSS_BUILD_DIR) / "tools/quake-to-tikz";

  if (!copy_file_and_report(quake_source_input_file, latex_quake_input_file)) {
    return -1;
  }

  if (int rc = convert_quake_to_tikz(
          quake_to_tikz_tool_path, latex_quake_input_file,
          latex_tikz_input_file, "./logs/passtest_to_tikz_before.log");
      rc != 0) {
    return -1;
  }

  // Load, run pass pipeline, write out
  std::string quake_module_text =
      readFileToString(quake_source_input_file.string());
  auto [mlir_module, context_ptr] = extractMLIRContext(quake_module_text);
  MLIRContext &context = *context_ptr;
  mlir::PassManager pass_manager(&context);
  pass_manager.addPass(std::move(pass));
  pass_manager.addPass(mlir::createCanonicalizerPass());
  pass_manager.addPass(mlir::createCSEPass());
  if (mlir::failed(pass_manager.run(mlir_module))) {
    std::cerr << "\nThe pass failed for " << quake_source_input_file.string()
              << std::endl;
    return -1;
  }
  if (int rc = write_to_file(mlir_module, latex_quake_output_file); rc != 0) {
    return -1;
  }

  if (int rc = convert_quake_to_tikz(
          quake_to_tikz_tool_path, latex_quake_output_file,
          latex_tikz_output_file, "./logs/passtest_to_tikz_after.log");
      rc != 0) {
    return -1;
  }

  // Produce PNGs
  if (int rc = build_png_from_tikz_file(latex_tikz_input_file); rc != 0) {
    return -1;
  }
  if (int rc = build_png_from_tikz_file(latex_tikz_output_file); rc != 0) {
    return -1;
  }
  return 0;
}
} // namespace ai_pass_selector