#include <gtest/gtest.h>

#include <Utils/info_utils.hpp>

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

TEST(AIInfoUtilsTest, ResolveQuakeCircuitFromSearchResult) {
  const fs::path quake_file =
      fs::path(MQSS_TEST_SOURCE_DIR) / "tests/quake/ZeroRxToIdPass.qke";
  auto result = ai_pass_selector::prepare_circuit_input_path(
      quake_file.parent_path(), quake_file.stem().string(),
      quake_file.extension().string());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), quake_file);
}

TEST(AIInfoUtilsTest, RejectsUnsupportedExtensions) {
  const fs::path invalid_file =
      fs::path(MQSS_TEST_SOURCE_DIR) / "tests/quake/ZeroRxToIdPass.qke";
  auto result = ai_pass_selector::prepare_circuit_input_path(
      invalid_file.parent_path(), invalid_file.stem().string(), ".txt");
  EXPECT_FALSE(result.has_value());
}

TEST(AIInfoUtilsTest, RejectsQasmAndHintsConversion) {
  const fs::path qasm_file =
      fs::path(MQSS_TEST_SOURCE_DIR) / "tests/qasm/test-parser.qasm";
  testing::internal::CaptureStderr();
  auto result = ai_pass_selector::prepare_circuit_input_path(
      qasm_file.parent_path(), qasm_file.stem().string(),
      qasm_file.extension().string());
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(result.has_value());
  EXPECT_NE(stderr_output.find(".qke"), std::string::npos);
}
