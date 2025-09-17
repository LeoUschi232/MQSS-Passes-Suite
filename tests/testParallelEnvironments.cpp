#include "Torch/parallel_environments.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>

namespace fs = std::filesystem;

TEST(ParallelEnvironmentsTest, RegisterQuantumCircuitRejectsInvalidIndex) {
  ai_pass_selector::ParallelEnvironments environments(
      /*nr_environments=*/1,
      /*max_qubits=*/1,
      /*max_instructions=*/1,
      /*max_depth=*/1,
      /*max_steps=*/1);

  EXPECT_THROW(
      environments.register_quantum_circuit(/*index=*/1, fs::path{}),
      std::out_of_range);
}
