#include <iostream>
#include <vector>

#include "verify_result.hpp"

void verify_data_reset(const std::vector<float>  in, const std::vector<float> out) {
  const auto m = in.size();

  for(auto i = 0; i < out.size(); i++) {
    if(in[i % m] != out[i]) {
      std::cout << "Failed" << std::endl;
      return;
    }
  }
  std::cout << "Passed" << std::endl;
}

void verify_stream_and_persistent(const std::vector<float>  in_persistent, const std::vector<float> in_streaming, const std::vector<float> out) {
  const auto m = in_persistent.size();

  for(auto i = 0; i < out.size(); i++) {
    if((in_persistent[i % m] + in_streaming[i]) != out[i]) {
      std::cout << "Failed" << std::endl;
      return;
    }
  }
  std::cout << "Passed" << std::endl;
}