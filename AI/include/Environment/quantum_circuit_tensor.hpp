#ifndef QUANTUM_CIRCUIT_TENSOR_HPP
#define QUANTUM_CIRCUIT_TENSOR_HPP

#include <array>
#include <cassert>
#include <cstddef>
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
constexpr int NR_GATES = SUPPORTED_GATES.size();

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


// The gate parameters are whether the gate is adjoint up to three possible
// angles of unitary and rotation gates, making up to four parameters.
constexpr int MAX_GATE_PARAMS = 4;
constexpr int IS_CONTROL = 1;


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
    assert(0<=i && i<shape[0]
        && 0<=j && j<shape[1]);
    return quantum_circuit_data[i * shape[1] + j];
  }

  const T &operator()(int i, int j) const {
    return const_cast<InstructionBasedTensor &>(*this)(i, j);
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
    : shape{maxQubits, maxDepth,
            NR_GATES + MAX_GATE_PARAMS + IS_CONTROL + maxQubits},
      quantum_circuit_data(
          maxQubits * maxDepth *
          (NR_GATES + MAX_GATE_PARAMS + IS_CONTROL + maxQubits)) {
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

  T *raw() { return quantum_circuit_data.data(); }
  const T *raw() const { return quantum_circuit_data.data(); }
  std::size_t size() const { return quantum_circuit_data.size(); }
};

} // namespace ai_pass_selector
#endif // QUANTUM_CIRCUIT_TENSOR_HPP