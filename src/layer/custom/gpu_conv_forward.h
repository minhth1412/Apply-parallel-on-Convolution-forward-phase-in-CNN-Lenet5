#ifndef SRC_LAYER_CUSTOM_GPU_CONV_FORWARD_H
#define SRC_LAYER_CUSTOM_GPU_CONV_FORWARD_H
#include <cuda_runtime.h>

class GPU_Conv_Forward
{
public:
    // Except out_data, all the other data are constant and will not be changed when calling this function.
    void execute(const float* in_data, float* out_data, const float* weight_data,
        const int n, const int out_channel, const int in_channel,
        const int height_in, const int width_in, const int kernel_height);
};

#endif // SRC_LAYER_CUSTOM_GPU_CONV_FORWARD_H