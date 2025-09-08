#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <random>
#include <iostream>

int main() {
  tensorflow::Tensor t(tensorflow::DT_FLOAT, tensorflow::TensorShape({2, 3}));

  // Fill with random values using the host CPU.
  std::mt19937 gen(12345);
  std::uniform_real_distribution dist(0.0f, 1.0f);
  auto m = t.matrix<float>();
  for (int i = 0; i < m.dimension(0); ++i) {
    for (int j = 0; j < m.dimension(1); ++j) {
      m(i, j) = dist(gen);
    }
  }

  // Print a summary and the full values
  std::cout << "Tensor summary: " << t.SummarizeValue(6) << "\n";
  std::cout << "Tensor debug: " << t.DebugString() << "\n";
  return 0;
}