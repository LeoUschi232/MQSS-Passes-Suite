#include "Utils/progress_bar.hpp"

#include <iostream>
#include <string>

namespace ai_pass_selector {
void updateProgress(const int current, const int total,
                    const std::string &display_message) {
  constexpr int barWidth = 50;
  const float progress = static_cast<float>(current) / total;

  std::cout << "\r: [\033[32m";
  const int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }

  std::cout << "\033[0m] " << current << "/" << total << " ("
      << static_cast<int>(progress * 100.0) << "%) (" << display_message
      << ") \033[K";
  std::cout.flush();
}
} // namespace ai_pass_selector