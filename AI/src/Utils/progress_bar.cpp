#include "Utils/progress_bar.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace ai_pass_selector {
void updateProgress(const int current, const int total,
                    const std::string &display_message) {
  constexpr int bar_width = 50;
  const float progress = static_cast<float>(current) / total;

  std::cout << "\r: [\033[32m";
  const int position = bar_width * progress;
  for (int i = 0; i < bar_width; ++i) {
    if (i < position) {
      std::cout << "=";
    } else if (i == position) {
      std::cout << ">";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "\033[0m] " << current << "/" << total << " ("
            << static_cast<int>(progress * 100.0) << "%) (" << display_message
            << ") \033[K";
  std::cout.flush();
}

void updateProgresses(const std::vector<std::pair<int, int>> &progresses,
                      const std::string &display_message) {
  if (progresses.empty()) {
    return;
  }
  constexpr int bar_width = 50;
  std::ostringstream progress_stream;
  auto [first_progress, first_total] = progresses[0];
  const float main_progress = static_cast<float>(first_progress) / first_total;
  progress_stream << std::to_string(first_progress) << "/"
                  << std::to_string(first_total);
  for (unsigned int i = 1; i < progresses.size(); i++) {
    auto [progress, total] = progresses[i];
    progress_stream << ", " << std::to_string(progress) << "/"
                    << std::to_string(total);
  }
  std::cout << "\r: [\033[32m";
  const int position = bar_width * main_progress;
  for (int i = 0; i < bar_width; ++i) {
    if (i < position) {
      std::cout << "=";
    } else if (i == position) {
      std::cout << ">";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "\033[0m] " << progress_stream.str() << " ("
            << static_cast<int>(main_progress * 100.0) << "%) ("
            << display_message << ") \033[K";
  std::cout.flush();
}
} // namespace ai_pass_selector