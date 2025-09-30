#ifndef QUANTUM_CIRCUIT_TENSOR_HPP
#define QUANTUM_CIRCUIT_TENSOR_HPP

// Standard library includes
#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace std::literals;

namespace ai_pass_selector {
// C++17 replacement for std::to_array
template <class T, class... U>
constexpr std::array<T, sizeof...(U)> make_array(U &&...u) {
  return {{static_cast<T>(u)...}};
}

// ---- Supported gates must mirror the gate map ----
constexpr auto SUPPORTED_GATES = make_array<std::string_view>(
    "x"sv, "y"sv, "z"sv, "h"sv, "s"sv, "t"sv, "rx"sv, "ry"sv, "rz"sv, "swap"sv,
    "r1"sv, "u2"sv, "u3"sv, "phased_rx"sv, "mx"sv, "my"sv, "mz"sv);
constexpr unsigned int NR_GATES = SUPPORTED_GATES.size();
constexpr unsigned int MAX_GATE_PARAMS = 3;

constexpr int GATE_INDEX(std::string_view gate) {
  for (int i = 0; i < NR_GATES; ++i) {
    if (SUPPORTED_GATES[i] == gate) {
      return i;
    }
  }
  return -1;
}

constexpr std::array<double, NR_GATES> GATE_ONE_HOT(std::string_view gate) {
  std::array<double, NR_GATES> one_hot{};
  one_hot[GATE_INDEX(gate)] = 1.0;
  return one_hot;
}

inline std::vector<double> index_to_one_hot(unsigned int size,
                                            unsigned int index) {
  std::vector one_hot(size, 0.0);
  if (index < size) {
    one_hot[index] = 1.0;
  }
  return one_hot;
}

inline std::vector<double>
index_to_one_hot(unsigned int size, const std::vector<unsigned int> &indexes) {
  std::vector multi_hot(size, 0.0);
  if (indexes.size() <= 0) {
    return multi_hot;
  }
  for (unsigned int index : indexes) {
    if (index < size) {
      multi_hot[index] = 1.0;
    }
  }
  return multi_hot;
}

constexpr unsigned int
MAX_QUBITS_TO_INSTRUCTION_REPRESENTATION_SIZE(unsigned int max_qubits) {
  return NR_GATES + MAX_GATE_PARAMS + max_qubits;
}

template <class T> struct InstructionsTensor {
  std::array<unsigned int, 2> shape{};
  std::vector<T> quantum_circuit_data;

  // Controls qubits triggered negative.
  // Target qubits triggered positive.
  explicit InstructionsTensor(unsigned int max_qubits)
      : shape{0, MAX_QUBITS_TO_INSTRUCTION_REPRESENTATION_SIZE(max_qubits)} {}

  void reserve(unsigned int nr_instructions) {
    quantum_circuit_data.reserve(nr_instructions * shape[1]);
  }

  void append(const std::vector<T> &values) {
    if (values.size() != shape[1]) {
      throw std::runtime_error(
          "Instruction feature size does not match tensor shape.");
    }
    shape[0]++;
    quantum_circuit_data.insert(quantum_circuit_data.end(), values.begin(),
                                values.end());
  }

  void replace(unsigned int i, const std::vector<T> &values) {
    if (values.size() != shape[1]) {
      throw std::runtime_error(
          "Instruction feature size does not match tensor shape.");
    }
    if (i >= shape[0]) {
      throw std::runtime_error("Instruction index out of range.");
    }
    std::copy(values.begin(), values.end(),
              quantum_circuit_data.begin() + i * shape[1]);
  }

  T &operator()(unsigned int i, unsigned int j) {
    assert(i < shape[0] && j < shape[1]);
    return quantum_circuit_data[i * shape[1] + j];
  }
  const T &operator()(unsigned int i, unsigned int j) const {
    assert(i < shape[0] && j < shape[1]);
    return quantum_circuit_data[i * shape[1] + j];
  }

  T *raw() { return quantum_circuit_data.data(); }
  const T *raw() const { return quantum_circuit_data.data(); }
  std::size_t size() const { return quantum_circuit_data.size(); }
};

} // namespace ai_pass_selector
#endif // QUANTUM_CIRCUIT_TENSOR_HPP
