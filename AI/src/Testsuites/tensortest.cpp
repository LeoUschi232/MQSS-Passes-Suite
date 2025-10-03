#include "Testsuites/tensortest.hpp"

// Environment includes
#include "Environment/quantum_circuit_environment.hpp"

// Passes includes
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"

// Utils includes
#include "Utils/dataset_conversion.hpp"
#include "Utils/progress_bar.hpp"
#include "Utils/tensor_utils.hpp"

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
// ------------------------------------------------------------
// Tensortest → TikZ PNGs (refactored, same behavior)
// ------------------------------------------------------------
void convertAllTensortestCircuitsToTikz() {
  std::cout << "Converting quake tensortest circuits to tikz." << std::endl;
  try {
    fs::create_directories("./logs");
    std::string cmd = "echo \"tensortest_to_tikz_before.log:\n\" > "
                      "./logs/tensortest_to_tikz_before.log";
    std::system(cmd.c_str());
    cmd = "echo \"tensortest_to_tikz_after.log:\n\" > "
          "./logs/tensortest_to_tikz_after.log";
    std::system(cmd.c_str());
  } catch (const fs::filesystem_error &e) {
    std::cerr << "\nError creating directory: " << e.what() << std::endl;
    return;
  }
  unsigned int nrTensortestCircuits = 0;
  while (fs::exists(fs::path(AI_DATASET_DIR) / "Quake/Tensortest" /
                    ("tensortest" + std::to_string(nrTensortestCircuits + 1) +
                     "_input.qke"))) {
    nrTensortestCircuits++;
  }

  for (unsigned int current = 1; current <= nrTensortestCircuits; current++) {
    updateProgress(current - 1, nrTensortestCircuits,
                   "tensortest" + std::to_string(current));
    const pid_t child_pid = fork();
    if (child_pid == 0) {
      exit(convertTensortestCircuitToTikz(current));
    }
    int status;
    waitpid(child_pid, &status, 0);
    if (WIFSIGNALED(status)) {
      std::cerr << "\nCircuit: tensortest" << current << " crashed with signal "
                << strsignal(WTERMSIG(status)) << std::endl;
      break;
    }
    updateProgress(current, nrTensortestCircuits, "");
  }

  const fs::path latex_dir = fs::path(AI_DATASET_DIR) / "Latex";
  const fs::path tex_file = latex_dir / "tensortest_quantum_circuits.tex";
  const std::string compile_pdf =
      "pdflatex -interaction=nonstopmode -halt-on-error --shell-escape "
      "-output-directory=" +
      latex_dir.string() + " " + tex_file.string() + " > /dev/null 2>&1";
  std::cout << "\nCompiling LaTeX file to PDF." << std::endl;
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
  const fs::path quake_source_input_file = fs::path(AI_DATASET_DIR) /
                                           "Quake/Tensortest" /
                                           (circuit_name + "_input.qke");
  const fs::path latex_tensortest_dir =
      fs::path(AI_DATASET_DIR) / "Latex/Tensortest";
  const fs::path latex_quake_input_file =
      latex_tensortest_dir / (circuit_name + "_input.qke");
  const fs::path latex_quake_output_file =
      latex_tensortest_dir / (circuit_name + "_output.qke");
  const fs::path latex_tikz_input_file =
      latex_tensortest_dir / (circuit_name + "_input.tikz");
  const fs::path latex_tikz_output_file =
      latex_tensortest_dir / (circuit_name + "_output.tikz");

  const fs::path quake_to_tikz_tool_path =
      fs::path(MQSS_BUILD_DIR) / "tools/quake-to-tikz";

  // Input quake → tikz
  if (!copy_file_and_report(quake_source_input_file, latex_quake_input_file)) {
    return -1;
  }
  if (int rc = convert_quake_to_tikz(
          quake_to_tikz_tool_path, latex_quake_input_file,
          latex_tikz_input_file, "./logs/tensortest_to_tikz_before.log");
      rc != 0) {
    return -1;
  }

  // Build instruction/depth observations and reconstruct two modules
  QuantumCircuitEnvironment quantum_circuit_environment(TENSORTEST_MAX_QUBITS,
                                                        0);
  if (!quantum_circuit_environment.register_quantum_circuit(
          quake_source_input_file)) {
    return -1;
  }

  InstructionsTensor<double> observation =
      quantum_circuit_environment.get_observation();

  auto [reconstructed_from_tensor, ctx_instr] =
      recreateQuantumCircuitFromTensor(observation);

  if (int rc = write_module_to_file(reconstructed_from_tensor,
                                    latex_quake_output_file);
      rc != 0) {
    return -1;
  }

  // Reconstructed quake → tikz (two variants)
  if (int rc = convert_quake_to_tikz(
          quake_to_tikz_tool_path, latex_quake_output_file,
          latex_tikz_output_file, "./logs/tensortest_to_tikz_after.log");
      rc != 0) {
    return -1;
  }

  // Produce PNGs for input and two outputs
  if (int rc = build_png_from_tikz_file(latex_tikz_input_file); rc != 0) {
    return -1;
  }
  if (int rc = build_png_from_tikz_file(latex_tikz_output_file); rc != 0) {
    return -1;
  }
  return 0;
}

} // namespace ai_pass_selector