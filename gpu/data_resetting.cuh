#ifndef DATA_RESETTING_CUH
#define DATA_RESETTING_CUH

void run_data_reset(const float*  in, float* out, const int n, const int m, const int device_id);

void run_data_reset_with_l2_persistence(const float*  in, float* out, const int n, const int m, const int device_id);

void run_stream_and_persistent(const float*  in_persistent, const float* in_streaming, float* out, const int n, const int m, const int device_id);

void run_stream_and_persistent_with_l2_persistence(const float*  in_persistent, const float* in_streaming, float* out, const int n, const int m, const int device_id);

void run_stream_and_persistent_batched(const float*  in_persistent, const float* in_streaming, float* out, const int n, const int m, const int device_id);

#endif // DATA_RESETTING_CUH