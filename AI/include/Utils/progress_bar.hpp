#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

namespace ai_pass_selector {
/**
 * Function to display a tqdm-like progress bar in the console.
 * @param current Current progress value.
 * @param total Total value to reach.
 * @param display_message Message to display alongside the progress bar.
 */
void updateProgress(int current, int total,
                    const std::string &display_message);
} // namespace ai_pass_selector

#endif // PROGRESS_BAR_HPP