#include "Utils/conversion_workflow.hpp"

// MLIR includes
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

// Passes includes
#include "Passes/Cancellations.hpp"
#include "Passes/Decompositions.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"

// Utils includes
#include "Utils/progress_bar.hpp"

// Stdandard library includes
#include "Environment/environment.hpp"
#include "Utils/tensor_utils.hpp"

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
using namespace mqss::support::quakeDialect;

namespace ai_pass_selector {
// -----------------------------
// Small, file-local utilities.
// -----------------------------

int run_shell_command(const std::string &command, const std::string &task) {
  int return_code = 0;
  try {
    return_code = std::system(command.c_str());
  } catch (const std::exception &e) {
    std::cerr << "\nException running " << task << ": " << e.what() <<
        std::endl;
    return -1;
  }
  if (return_code != 0) {
    std::cerr << "\n" << task
        << " failed (return code: " << return_code << ")\n";
    return return_code;
  }
  return 0;
}

bool copy_file_and_report(const fs::path &source, const fs::path &destination) {
  std::error_code error_code;
  fs::create_directories(destination.parent_path(), error_code);
  if (!fs::copy_file(source, destination, fs::copy_options::overwrite_existing,
                     error_code)) {
    std::cerr << "\nFailed to copy " << source.string()
        << " to " << destination.string()
        << (error_code ? ": " + error_code.message() : "") << std::endl;
    return false;
  }
  return true;
}

int convert_quake_to_tikz(
    const fs::path &quake_to_tikz_tool_path, const fs::path &quake_input_path,
    const fs::path &tikz_output_path, const std::string &append_to_log_file) {
  const std::string command_line =
      quake_to_tikz_tool_path.string() +
      " --input " + quake_input_path.string() +
      " --output " + tikz_output_path.string() +
      " >> " + append_to_log_file;
  return run_shell_command(
      command_line,
      "Conversion qke → tikz for " + quake_input_path.string());
}

int write_module_to_file(ModuleOp module,
                         const fs::path &destination_file_path) {
  std::string module_serialized_text;
  llvm::raw_string_ostream string_stream(module_serialized_text);
  module->print(string_stream);

  std::ofstream output_file(destination_file_path.string());
  if (!output_file) {
    std::cerr << "\nFailed to open " << destination_file_path.string() <<
        std::endl;
    return -1;
  }
  output_file << module_serialized_text;
  output_file.close();
  return 0;
}

int build_png_from_tikz_file(const fs::path &tikz_file_path) {
  // Minimal wrapper document (standalone) that \input{sometikz.tikz}
  // Write temp.tex next to CWD (consistent with existing workflow).
  {
    const std::string latex_wrapper =
        "\\documentclass{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\usetikzlibrary{quantikz}\n"
        "\\begin{document}\n"
        "\\input{" + tikz_file_path.string() + "}\n"
        "\\end{document}\n";
    std::ofstream temp_tex_file("temp.tex");
    if (!temp_tex_file) {
      std::cerr << "\nFailed to create temp.tex for " << tikz_file_path <<
          std::endl;
      return -1;
    }
    temp_tex_file << latex_wrapper;
  }

  const std::string png_output_base =
      tikz_file_path.string().substr(0, tikz_file_path.string().find(".tikz"));

  // Compile LaTeX and convert to PNG (trimmed). Preserve your flags & quieting.
  const std::string command_line =
      "pdflatex -interaction=nonstopmode temp.tex > /dev/null 2>&1 && "
      "convert -density 300 -strip temp.pdf -trim -quality 90 " +
      png_output_base + ".png > /dev/null 2>&1 && "
      "rm temp.* > /dev/null 2>&1";

  return run_shell_command(
      command_line, "PNG conversion for " + tikz_file_path.string());
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


// ------------------------------------------------------------
// Passtest → TikZ PNGs
// ------------------------------------------------------------
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

  const int total = static_cast<int>(passes.size());
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
      "pdflatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory="
      + latex_dir.string() + " " + tex_file.string() + " > /dev/null 2>&1";
  std::cout << "Compiling LaTeX file to PDF." << std::endl;
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
      quake_to_tikz_tool_path,
      latex_quake_input_file,
      latex_tikz_input_file,
      "./logs/passtest_to_tikz_before.log"); rc != 0) {
    return -1;
  }

  // Load, run pass pipeline, write out
  std::string quake_module_text = readFileToString(
      quake_source_input_file.string());
  auto [mlir_module, context_ptr] = extractMLIRContext(quake_module_text);
  MLIRContext &context = *context_ptr;
  mlir::PassManager pass_manager(&context);
  pass_manager.addPass(std::move(pass));
  pass_manager.addPass(mlir::createCanonicalizerPass());
  pass_manager.addPass(mlir::createCSEPass());
  if (mlir::failed(pass_manager.run(mlir_module))) {
    std::cerr << "\nThe pass failed for " << quake_source_input_file.string() <<
        std::endl;
    return -1;
  }
  if (int rc = write_module_to_file(mlir_module, latex_quake_output_file);
    rc != 0) {
    return -1;
  }

  if (int rc = convert_quake_to_tikz(
      quake_to_tikz_tool_path,
      latex_quake_output_file,
      latex_tikz_output_file,
      "./logs/passtest_to_tikz_after.log"); rc != 0) {
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


// ------------------------------------------------------------
// Tensortest → TikZ PNGs (refactored, same behavior)
// ------------------------------------------------------------
void convertAllTensortestCircuitsToTikz(int nrTensortestCircuits) {
  std::cout << "Converting " << nrTensortestCircuits
      << " quake tensortest circuits to tikz." << std::endl;
  if (nrTensortestCircuits == 0) {
    std::cout << "No circuits found for conversion to TikZ." << std::endl;
    return;
  }
  try {
    fs::create_directories("./logs");
    std::string cmd =
        "echo \"tensortest_to_tikz_before.log:\n\" > ./logs/tensortest_to_tikz_before.log";
    std::system(cmd.c_str());
    cmd =
        "echo \"tensortest_to_tikz_after1.log:\n\" > ./logs/tensortest_to_tikz_after1.log";
    std::system(cmd.c_str());
    cmd =
        "echo \"tensortest_to_tikz_after2.log:\n\" > ./logs/tensortest_to_tikz_after2.log";
    std::system(cmd.c_str());
  } catch (const fs::filesystem_error &e) {
    std::cerr << "\nError creating directory: " << e.what() << std::endl;
    return;
  }

  for (int current = 1; current <= nrTensortestCircuits; current++) {
    updateProgress(current - 1, nrTensortestCircuits,
                   "tensortest" + std::to_string(current));
    const pid_t child_pid = fork();
    if (child_pid == 0) {
      exit(convertTensortestCircuitToTikz(current));
    }
    int status;
    waitpid(child_pid, &status, 0);
    if (WIFSIGNALED(status)) {
      std::cerr << "\nCircuit: tensortest" << current
          << " crashed with signal " << strsignal(WTERMSIG(status))
          << std::endl;
      break;
    }
    updateProgress(current, nrTensortestCircuits, "");
  }

  const fs::path latex_dir = fs::path(AI_DATASET_DIR) / "Latex";
  const fs::path tex_file = latex_dir / "tensortest_quantum_circuits.tex";
  const std::string compile_pdf =
      "pdflatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory="
      + latex_dir.string() + " " + tex_file.string() + " > /dev/null 2>&1";
  std::cout << "Compiling LaTeX file to PDF." << std::endl;
  if (const int ret = std::system(compile_pdf.c_str()); ret != 0) {
    std::cerr << "\nPDF generation failed for tensortest_quantum_circuits.tex "
        << "(return code: " << ret << ")" << std::endl;
    return;
  }
  std::cout << "PDF generated successfully at "
      << (latex_dir / "tensortest_quantum_circuits.pdf").string()
      << std::endl;
}

int convertTensortestCircuitToTikz(int index) {
  std::string circuit_name = "tensortest" + std::to_string(index);
  const fs::path quake_source_input_file =
      fs::path(AI_DATASET_DIR) / "Quake/Tensortest" / (
        circuit_name + "_input.qke");

  const fs::path latex_tensortest_dir =
      fs::path(AI_DATASET_DIR) / "Latex/Tensortest";
  const fs::path latex_quake_input_file =
      latex_tensortest_dir / (circuit_name + "_input.qke");
  const fs::path latex_quake_output_file1 =
      latex_tensortest_dir / (circuit_name + "_output1.qke");
  const fs::path latex_quake_output_file2 =
      latex_tensortest_dir / (circuit_name + "_output2.qke");

  const fs::path latex_tikz_input_file =
      latex_tensortest_dir / (circuit_name + "_input.tikz");
  const fs::path latex_tikz_output_file1 =
      latex_tensortest_dir / (circuit_name + "_output1.tikz");
  const fs::path latex_tikz_output_file2 =
      latex_tensortest_dir / (circuit_name + "_output2.tikz");

  const fs::path quake_to_tikz_tool_path =
      fs::path(MQSS_BUILD_DIR) / "tools/quake-to-tikz";

  // Input quake → tikz
  if (!copy_file_and_report(quake_source_input_file, latex_quake_input_file)) {
    return -1;
  }
  if (int rc = convert_quake_to_tikz(
      quake_to_tikz_tool_path,
      latex_quake_input_file,
      latex_tikz_input_file,
      "./logs/tensortest_to_tikz_before.log"); rc != 0) {
    return -1;
  }

  // Build instruction/depth observations and reconstruct two modules
  std::string quake_module_text = readFileToString(
      quake_source_input_file.string());
  auto [input_module, ctx_ptr] = extractMLIRContext(quake_module_text);

  QuantumCircuitEnviorment quantum_circuit_enviorment(
      TENSORTEST_MAX_QUBITS, TENSORTEST_MAX_INSTRUCTIONS,
      TENSORTEST_MAX_DEPTH, input_module);
  std::cout << "\nBlock 1" << std::endl;

  InstructionBasedTensor<double> instruction_based_observation
      = quantum_circuit_enviorment.get_instruction_based_observation();
  std::cout << "\nBlock 2" << std::endl;

  DepthBasedTensor<double> depth_based_observation
      = quantum_circuit_enviorment.get_depth_based_observation();
  std::cout << "\nBlock 3" << std::endl;

  ModuleOp reconstructed_from_instruction_tensor =
      recreateQuantumCircuitFromInstructionBasedTensor(
          instruction_based_observation);
  std::cout << "\nBlock 4" << std::endl;

  ModuleOp reconstructed_from_depth_tensor =
      recreateQuantumCircuitFromDepthBasedTensor(depth_based_observation);
  std::cout << "\nBlock 5" << std::endl;

  if (int rc = write_module_to_file(
      reconstructed_from_instruction_tensor,
      latex_quake_output_file1); rc != 0) {
    return -1;
  }
  std::cout << "\nBlock 6" << std::endl;

  if (int rc = write_module_to_file(
      reconstructed_from_depth_tensor, latex_quake_output_file2); rc != 0) {
    return -1;
  }
  std::cout << "\nBlock 7" << std::endl;

  // Reconstructed quake → tikz (two variants)
  if (int rc = convert_quake_to_tikz(
      quake_to_tikz_tool_path,
      latex_quake_output_file1,
      latex_tikz_output_file1,
      "./logs/tensortest_to_tikz_after1.log"); rc != 0) {
    return -1;
  }
  if (int rc = convert_quake_to_tikz(
      quake_to_tikz_tool_path,
      latex_quake_output_file2,
      latex_tikz_output_file2,
      "./logs/tensortest_to_tikz_after2.log"); rc != 0) {
    return -1;
  }

  // Produce PNGs for input and two outputs
  if (int rc = build_png_from_tikz_file(latex_tikz_input_file); rc != 0) {
    return -1;
  }
  if (int rc = build_png_from_tikz_file(latex_tikz_output_file1); rc != 0) {
    return -1;
  }
  if (int rc = build_png_from_tikz_file(latex_tikz_output_file2); rc != 0) {
    return -1;
  }
  return 0;
}

} // namespace ai_pass_selector