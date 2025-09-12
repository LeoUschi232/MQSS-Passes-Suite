#ifndef QUANTUM_CIRCUIT_TENSOR_HPP
#define QUANTUM_CIRCUIT_TENSOR_HPP

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
constexpr std::array<T, sizeof...(U)> make_array(U &&... u) {
  return {{static_cast<T>(u)...}};
}


// ---- Supported gates must mirror the gate map ----
constexpr auto SUPPORTED_GATES = make_array<std::string_view>(
    "x"sv, "y"sv, "z"sv, "h"sv, "s"sv, "t"sv, "rx"sv, "ry"sv,
    "rz"sv, "swap"sv, "r1"sv, "u2"sv, "u3"sv, "phased_rx"sv, "mx"sv, "my"sv,
    "mz"sv
    );


// The gate parameters are whether the gate is adjoint up to three possible
// angles of unitary and rotation gates, making up to four parameters.
constexpr int NR_GATES = SUPPORTED_GATES.size();
constexpr int MAX_GATE_ANGLES = 3;
constexpr int MAX_GATE_PARAMS = 4;
constexpr int QUBIT_ROLE_PARAMS = 2;

constexpr int GATE_INDEX(std::string_view gate) {
  for (int i = 0; i < NR_GATES; ++i) {
    if (SUPPORTED_GATES[i] == gate) {
      return i;
    }
  }
  return -1; // not found
}

constexpr std::array<double, NR_GATES> GATE_ONE_HOT(std::string_view gate) {
  std::array<double, NR_GATES> one_hot{};
  if (const int idx = GATE_INDEX(gate); idx >= 0) {
    one_hot[static_cast<std::size_t>(idx)] = 1.0;
  }
  return one_hot;
}

inline std::vector<double> index_to_one_hot(int size, int index) {
  std::vector one_hot(size, 0.0);
  if (0 <= index && index < size) {
    one_hot[index] = 1.0;
  }
  return one_hot;
}

inline std::vector<double> index_to_one_hot(
    int size, const std::vector<int> &indexes) {
  std::vector multi_hot(size, 0.0);
  if (indexes.size() <= 0) {
    return multi_hot;
  }
  for (int index : indexes) {
    if (0 <= index && index < size) {
      multi_hot[index] = 1.0;
    }
  }
  return multi_hot;
}


template <class T>
struct InstructionBasedTensor {
  std::array<int, 2> shape{};
  std::vector<T> quantum_circuit_data;

  // Number of qubits if multiplied by two because each qubit mut be able to be
  // triggered twice per instruction.
  // Qubits triggered in the first set are controls.
  // Qubits triggered in the second set are targets.
  InstructionBasedTensor(int maxQubits, int maxInstructions)
    : shape{maxInstructions, 2 * maxQubits + NR_GATES + MAX_GATE_PARAMS},
      quantum_circuit_data(
          maxInstructions * (2 * maxQubits + NR_GATES + MAX_GATE_PARAMS)) {
  }

  T &operator()(int i, int j) {
    assert(0<=i && i<shape[0] && 0<=j && j<shape[1]);
    return quantum_circuit_data[i * shape[1] + j];
  }

  const T &operator()(int i, int j) const {
    return const_cast<InstructionBasedTensor &>(*this)(i, j);
  }

  void fillInstructionFeature(
      int instruction_index, const std::vector<T> &values) {
    if (values.size() != static_cast<size_t>(shape[1])
        || instruction_index < 0 || instruction_index >= shape[0]) {
      throw std::runtime_error(
          "Instruction feature size does not match tensor shape.");
    }
    const int offset = instruction_index * shape[1];
    std::copy(values.begin(), values.end(),
              quantum_circuit_data.begin() + offset);
  }

  T *row_ptr(int i) {
    assert(0 <= i && i < shape[0]);
    return quantum_circuit_data.data() + i * shape[1];
  }

  void clear_row(int i) {
    std::fill_n(row_ptr(i), shape[1], T{});
  }

  T *raw() { return quantum_circuit_data.data(); }
  const T *raw() const { return quantum_circuit_data.data(); }
  std::size_t size() const { return quantum_circuit_data.size(); }
};


template <class T>
struct DepthBasedTensor {
  std::array<int, 3> shape{};
  std::vector<T> quantum_circuit_data;

  // Here a hypothetical grid is constructed of maxQubits x maxDepth dimensions.
  // Every block in this grid may be empty or may be a gate acting on a qubit.
  // This allow representing multiple gates/operations taking place at the same
  // maxDepth along a circuit on different qubits.
  // However, here multi-qubit gates need additional information such as
  DepthBasedTensor(int maxQubits, int maxDepth)
    : shape{maxDepth, maxQubits,
            NR_GATES + MAX_GATE_PARAMS + QUBIT_ROLE_PARAMS + maxQubits},
      quantum_circuit_data(
          maxDepth * maxQubits *
          (NR_GATES + MAX_GATE_PARAMS + QUBIT_ROLE_PARAMS + maxQubits)) {
  }

  T &operator()(int i, int j, int k) {
    assert(0 <= i && i < shape[0]
        && 0 <= j && j < shape[1]
        && 0 <= k && k < shape[ 2]);
    return quantum_circuit_data[(i * shape[1] + j) * shape[2] + k];
  }

  const T &operator()(int i, int j, int k) const {
    return const_cast<DepthBasedTensor &>(*this)(i, j, k);
  }

  T *cell_ptr(int depth_index, int qubit_index) {
    assert(0 <= depth_index && depth_index < shape[0]
        && 0 <= qubit_index && qubit_index < shape[1]);
    return quantum_circuit_data.data()
           + (depth_index * shape[1] + qubit_index) * shape[2];
  }


  T *raw() { return quantum_circuit_data.data(); }
  const T *raw() const { return quantum_circuit_data.data(); }
  std::size_t size() const { return quantum_circuit_data.size(); }
};

} // namespace ai_pass_selector
#endif // QUANTUM_CIRCUIT_TENSOR_HPP