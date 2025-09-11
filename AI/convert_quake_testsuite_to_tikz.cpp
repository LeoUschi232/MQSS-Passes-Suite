// Include the header relative to the include directory.
#include "Utils/conversion_workflow.hpp"

#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    return -1;
  }
  std::string testsuite = argv[1];
  std::transform(
      testsuite.begin(), testsuite.end(), testsuite.begin(),
      [](unsigned char c) { return std::tolower(c); });
  if (testsuite == "passtest") {
    ai_pass_selector::convertAllPasstestCircuitsToTikz();
  } else if (testsuite == "tensortest") {
    ai_pass_selector::convertAllTensortestCircuitsToTikz();
  } else {
    std::cout << "Unknown test suit: " << argv[1] << std::endl;
    return -1;
  }
  return 0;
}