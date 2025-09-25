#ifndef DATASET_CONVERSION_HPP
#define DATASET_CONVERSION_HPP

// MLIR includes
#include "mlir/IR/BuiltinOps.h"

// Stdandard library includes
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace ai_pass_selector {
/**
 *
 * @param command
 * @param task
 * @return
 */
int run_shell_command(const std::string &command, const std::string &task);

/**
 *
 * @param source
 * @param destination
 * @return
 */
bool copy_file_and_report(const fs::path &source, const fs::path &destination);

/**
 *
 * @param quake_to_tikz_tool_path
 * @param quake_input_path
 * @param tikz_output_path
 * @param append_to_log_file
 * @return
 */
int convert_quake_to_tikz(const fs::path &quake_to_tikz_tool_path,
                          const fs::path &quake_input_path,
                          const fs::path &tikz_output_path,
                          const std::string &append_to_log_file);

/**
 *
 * @param module
 * @param destination_file_path
 * @return
 */
int write_module_to_file(mlir::ModuleOp module,
                         const fs::path &destination_file_path);

/**
 *
 * @param tikz_file_path
 * @return
 */
int build_png_from_tikz_file(const fs::path &tikz_file_path);

/**
 * Convert all Qasm files in the AI/Dataset/Qasm directory to Quake files
 * int the AI/Dataset/Quake directory.
 */
void convertAllQasmDatasetsToQuake();

/**
 * Convert all Qasm files in a given subdirectory if AI/Dataset/Qasm to Quake
 * files inside AI/Dataset/Quake.
 * @param subdirectory The subdirectory inside AI/Dataset/Qasm to convert
 * the Qasm files form.
 * @return 0 on success, -1 on failure.
 */
int convertQasmDatasetToQuake(const std::string &subdirectory);
} // namespace ai_pass_selector

#endif // DATASET_CONVERSION_HPP