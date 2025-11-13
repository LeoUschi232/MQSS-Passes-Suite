#ifndef INFO_UTILS_HPP
#define INFO_UTILS_HPP

// Torch includes
#include <torch/torch.h>

// Support includes
#include "Interfaces/Constants.hpp"

// Standard Library includes
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>

namespace fs = std::filesystem;

namespace ai_pass_selector {
////////////////////////////////////////////////////////////////////////////////
/// Mersenne Twister RNG
inline std::mt19937 &qc_rng() {
  static std::mt19937 rng_engine{std::random_device{}()};
  return rng_engine;
}
inline void seed_qc_rng(uint32_t seed) {
  qc_rng().seed(seed);
}
inline double random01() {
  thread_local std::uniform_real_distribution dist01(0.0, 1.0);
  return dist01(qc_rng());
}
inline float randomAngle() {
  thread_local std::uniform_real_distribution<float> distAngle(0.0, 2.0 * PI);
  return distAngle(qc_rng());
}
inline int randomInt(int start, int end) {
  std::uniform_int_distribution distInt(start, end - 1);
  return distInt(qc_rng());
}
inline int randomPoisson(double mean) {
  if (mean <= 0.0) {
    return 0;
  }
  std::poisson_distribution distPoisson(mean);
  return std::max(0, distPoisson(qc_rng()));
}
////////////////////////////////////////////////////////////////////////////////
/// All no-nonsense quantum circuit should have at least 2 qubits and 2 gates.
constexpr unsigned int GLOBAL_MIN_NR_QUBITS = 2u;
constexpr unsigned int GLOBAL_MIN_NR_GATES = 2u;

struct PassSelectorRuntimeParam {
  using value_t =
      std::variant<int, double, bool, std::string, torch::DeviceType>;
  value_t value;

  /// Implicit Constructors
  PassSelectorRuntimeParam() = default;
  PassSelectorRuntimeParam(int x) : value(x) {}
  PassSelectorRuntimeParam(double x) : value(x) {}
  PassSelectorRuntimeParam(bool x) : value(x) {}
  PassSelectorRuntimeParam(std::string s) : value(std::move(s)) {}
  PassSelectorRuntimeParam(const char *s) : value(std::string(s)) {}
  PassSelectorRuntimeParam(torch::DeviceType device) : value(device) {}

  /// Convenience Checks
  bool is_int() const { return std::holds_alternative<int>(value); }
  bool is_double() const { return std::holds_alternative<double>(value); }
  bool is_bool() const { return std::holds_alternative<bool>(value); }
  bool is_string() const { return std::holds_alternative<std::string>(value); }
  bool is_device_type() const {
    return std::holds_alternative<torch::DeviceType>(value);
  }

  /// As value methods
  int to_int() const {
    if (is_int()) {
      return std::get<int>(value);
    }
    if (is_double()) {
      return static_cast<int>(std::get<double>(value));
    }
    if (is_bool()) {
      return std::get<bool>(value) ? 1 : 0;
    }
    if (is_string()) {
      try {
        return std::stoi(std::get<std::string>(value));
      } catch (const std::exception &_) {
        return 0;
      }
    }
    if (is_device_type()) {
      return static_cast<int>(std::get<torch::DeviceType>(value));
    }
    return 0;
  }
  double to_double() const {
    if (is_int()) {
      return std::get<int>(value);
    }
    if (is_double()) {
      return std::get<double>(value);
    }
    if (is_bool()) {
      return std::get<bool>(value) ? 1.0 : 0.0;
    }
    if (is_string()) {
      try {
        return std::stod(std::get<std::string>(value));
      } catch (const std::exception &_) {
        return 0.0;
      }
    }
    if (is_device_type()) {
      return static_cast<double>(std::get<torch::DeviceType>(value));
    }
    return 0;
  }
  bool to_bool() const {
    if (is_int()) {
      return std::get<int>(value) != 0;
    }
    if (is_double()) {
      return std::get<double>(value) != 0.0;
    }
    if (is_bool()) {
      return std::get<bool>(value);
    }
    if (is_string()) {
      return std::get<std::string>(value) == "true";
    }
    return false;
  }
  std::string to_string() const {
    if (is_int()) {
      return std::to_string(std::get<int>(value));
    }
    if (is_double()) {
      return std::to_string(std::get<double>(value));
    }
    if (is_bool()) {
      return std::get<bool>(value) ? "true" : "false";
    }
    if (is_string()) {
      return std::get<std::string>(value);
    }
    if (is_device_type()) {
      return c10::DeviceTypeName(std::get<torch::DeviceType>(value));
    }
    return "";
  }
  torch::DeviceType to_device_type() const {
    if (is_device_type()) {
      return std::get<torch::DeviceType>(value);
    }
    if (is_bool()) {
      return torch::kCPU;
    }
    int as_int = 0;
    if (is_int()) {
      as_int = std::get<int>(value);
    } else if (is_double()) {
      as_int = static_cast<int>(std::get<double>(value));
    } else if (is_string()) {
      auto device_name = std::get<std::string>(value);
      std::transform(device_name.begin(), device_name.end(),
                     device_name.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (device_name == "cuda") {
        return torch::kCUDA;
      }
      try {
        as_int = std::stoi(device_name);
      } catch (const std::exception &_) {
        return torch::kCPU;
      }
    }
    if (as_int < static_cast<int>(torch::DeviceType::CPU) ||
        as_int >= static_cast<int>(
                      torch::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)) {
      return torch::kCPU;
    }
    return static_cast<torch::DeviceType>(as_int);
  }

  /// Convenience Getters
  explicit operator int() const { return to_int(); }
  explicit operator double() const { return to_double(); }
  explicit operator bool() const { return to_bool(); }
  explicit operator std::string() const { return to_string(); }
  explicit operator torch::DeviceType() const { return to_device_type(); }
};
inline std::unordered_map<std::string, PassSelectorRuntimeParam> GLOBAL_PARAMS;

/**
 * @param a
 * @param b
 * @param atol
 * @return
 */
bool isclose(double a, double b, double atol = 1e-12);

/**
 * @param str
 * @param delimiter
 * @return
 */
std::vector<std::string> split_string(const std::string &str, char delimiter);

/**
 * Given a circuit name or its full filepath, search for the circuit and
 * returns its folder, name and extension.
 * @param circuit_path Name or full filepath of the circuit to search for.
 * @return
 */
std::optional<fs::path> search_circuit(const fs::path &circuit_path);

/**
 * @param circuit_path
 */
void print_circuit_info(fs::path circuit_path);

/**
 * @param dataset_name
 * @return
 */
std::vector<fs::path> get_dataset_files(const std::string &dataset_name);

/**
 * @param dataset_name
 * @return
 */
std::optional<std::vector<std::pair<std::string, std::string>>>
get_dataset_info(const std::string &dataset_name);

/**
 * @param dataset_name
 */
void print_dataset_info(const std::string &dataset_name);

/**
 * @param agent_name
 */
void print_agent_info(const std::string &agent_name);
} // namespace ai_pass_selector

#endif // INFO_UTILS_HPP
