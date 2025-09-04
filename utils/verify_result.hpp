#ifndef VERIFY_RESULT_HPP
#define VERIFY_RESULT_HPP

#include <vector>

void verify_data_reset(const std::vector<float>  in, const std::vector<float> out);

void verify_stream_and_persistent(const std::vector<float>  in_persistent, const std::vector<float>  in_streaming, const std::vector<float> out);


#endif // VERIFY_RESULT_HPP