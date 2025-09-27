#include "Testsuites/randomtest.hpp"

// Environment includes
#include "Environment/random_quantum_circuit_generator.hpp"

// Utils includes
#include "Utils/dataset_conversion.hpp"
#include "Utils/info_utils.hpp"
#include "Utils/progress_bar.hpp"

// Standard library includes
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace ai_pass_selector {
void createAndConvertRandomCircuitsToTikz(unsigned int nr_circuits) {
  std::cout << "Creating and converting quake randomtest circuits to tikz."
            << std::endl;
  try {
    fs::create_directories("./logs");
    std::string cmd = "echo \"randomtest_to_tikz_before.log:\n\" > "
                      "./logs/randomtest_to_tikz_before.log";
    std::system(cmd.c_str());
    cmd = "echo \"randomtest_to_tikz_after.log:\n\" > "
          "./logs/randomtest_to_tikz_after.log";
    std::system(cmd.c_str());
  } catch (const fs::filesystem_error &e) {
    std::cerr << "\nError creating directory: " << e.what() << std::endl;
    return;
  }
  if (fs::path dataset_dir = fs::path(AI_DATASET_DIR) / "Quake/Randomtest";
      !fs::exists(dataset_dir)) {
    try {
      fs::create_directories(dataset_dir);
    } catch (const fs::filesystem_error &e) {
      std::cerr << "\nError creating directory: " << e.what() << std::endl;
      return;
    }
  }
  if (fs::path dataset_dir = fs::path(AI_DATASET_DIR) / "Latex/Randomtest";
      !fs::exists(dataset_dir)) {
    try {
      fs::create_directories(dataset_dir);
    } catch (const fs::filesystem_error &e) {
      std::cerr << "\nError creating directory: " << e.what() << std::endl;
      return;
    }
  }
  for (unsigned int current = 1; current <= nr_circuits; current++) {
    updateProgress(current - 1, nr_circuits,
                   "randomtest" + std::to_string(current));
    const pid_t child_pid = fork();
    if (child_pid == 0) {
      seed_qc_rng(current);
      exit(createAndConvertOneRandomCircuitToTikz(current));
    }
    int status;
    waitpid(child_pid, &status, 0);
    if (WIFSIGNALED(status)) {
      std::cerr << "\nCircuit: randomtest" << current << " crashed with signal "
                << strsignal(WTERMSIG(status)) << std::endl;
      break;
    }
    updateProgress(current, nr_circuits, "");
  }

  const fs::path latex_dir = fs::path(AI_DATASET_DIR) / "Latex";
  const fs::path tex_file = latex_dir / "randomtest_quantum_circuits.tex";
  const std::string compile_pdf =
      "pdflatex -interaction=nonstopmode -halt-on-error --shell-escape "
      "-output-directory=" +
      latex_dir.string() + " " + tex_file.string() + " > /dev/null 2>&1";
  std::cout << "\nCompiling LaTeX file to PDF." << std::endl;
  if (const int ret = std::system(compile_pdf.c_str()); ret != 0) {
    std::cerr << "\nPDF generation failed for randomtest_quantum_circuits.tex "
              << "(return code: " << ret << ")" << std::endl;
    return;
  }
  std::cout << "PDF generated successfully at "
            << (latex_dir / "randomtest_quantum_circuits.pdf").string()
            << std::endl;
}

int createAndConvertOneRandomCircuitToTikz(int index) {
  fs::path quake_qke_circuit_filepath =
      fs::path(AI_DATASET_DIR) /
      ("Quake/Randomtest/randomtest" + std::to_string(index) + ".qke");
  fs::path latex_qke_circuit_filepath =
      fs::path(AI_DATASET_DIR) /
      ("Latex/Randomtest/randomtest" + std::to_string(index) + ".qke");
  fs::path latex_tikz_circuit_filepath =
      fs::path(AI_DATASET_DIR) /
      ("Latex/Randomtest/randomtest" + std::to_string(index) + ".tikz");
  const fs::path quake_to_tikz_tool_path =
      fs::path(MQSS_BUILD_DIR) / "tools/quake-to-tikz";

  auto [circuit, contextPtr] = random_quantum_circuit_from_yaml_statistics(
      fs::path(RQCG_STATISTICS_DIR) / "RandomtestStatistics.yaml");
  if (int rc = write_module_to_file(circuit, quake_qke_circuit_filepath);
      rc != 0) {
    return -1;
  }
  if (!copy_file_and_report(quake_qke_circuit_filepath,
                            latex_qke_circuit_filepath)) {
    return -1;
  }
  if (int rc = convert_quake_to_tikz(
          quake_to_tikz_tool_path, latex_qke_circuit_filepath,
          latex_tikz_circuit_filepath, "./logs/randomtest_to_tikz_after.log");
      rc != 0) {
    return -1;
  }
  if (int rc = build_png_from_tikz_file(latex_tikz_circuit_filepath); rc != 0) {
    return -1;
  }
  return 0;
}
} // namespace ai_pass_selector