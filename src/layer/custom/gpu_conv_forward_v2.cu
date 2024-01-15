#include "gpu_conv_forward.h"
#include "gpu_utils.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH_LAYER_1 16
#define TILE_WIDTH_LAYER_3 12

__global__ void conv_forward_kernel(float* d_out, const float* d_in, const float* kernel,
    const int out_channel, const int in_channel,
    const int height, const int width, const int kernel_size)
{
    int TILE_WIDTH = (in_channel == 1) ? TILE_WIDTH_LAYER_1 : TILE_WIDTH_LAYER_3;
    const int INPUT_TILE_WIDTH = TILE_WIDTH + kernel_size - 1;

    const int height_out = height - kernel_size + 1;
    const int width_out = width - kernel_size + 1;

    int width_grid = (width_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int tile_h = (bz / width_grid) * TILE_WIDTH;
    int tile_w = (bz % width_grid) * TILE_WIDTH;

    int row_idx = tile_h + threadIdx.y; // row of the image matrix
    int col_idx = tile_w + threadIdx.x; // col of the image matrix

    extern __shared__ float s_input[];

#pragma unroll
    for (int c = 0; c < in_channel; c++)
    {
#pragma unroll
        for (int i = ty; i < INPUT_TILE_WIDTH; i += TILE_WIDTH)
        {
#pragma unroll
            for (int j = tx; j < INPUT_TILE_WIDTH; j += TILE_WIDTH)
            {
                if (tile_h + i < height && tile_w + j < width)
                {
                    s_input[c * INPUT_TILE_WIDTH * INPUT_TILE_WIDTH + i * INPUT_TILE_WIDTH + j]
                        = d_in[bx * in_channel * height * width + c * height * width + (tile_h + i) * width + tile_w + j];
                }
            }
        }
    }

    float accumulator = 0.0f;

    if (row_idx < height_out && col_idx < width_out)
    {
        for (int in_idx = 0; in_idx < in_channel; in_idx++)
        {
            for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++)
            {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++)
                {
                    int input_row = row_idx + kernel_row;
                    int input_col = col_idx + kernel_col;
                    int kernel_area = kernel_size * kernel_size;
                    int in_area = height * width;

                    accumulator += s_input[(bx * (in_channel * in_area)) + (in_idx * in_area) + (input_row * width) + input_col] *
                        kernel[(by * (in_channel * kernel_area)) + (in_idx * kernel_area) + (kernel_row * kernel_size) + kernel_col];
                }
            }
        }
        int out_area = height_out * width_out;
        d_out[bx * (out_channel * out_area) + by * out_area + row_idx * width_out + col_idx] = accumulator;
    }
}

void GPU_Conv_Forward::execute(const float* in_data, float* out_data, const float* weight_data,
    const int n, const int in_channel, const int out_channel,
    const int height_in, const int width_in, const int kernel_height)
{
    int TILE_WIDTH = (in_channel == 1) ? TILE_WIDTH_LAYER_1 : TILE_WIDTH_LAYER_3;
    printf("Optimization version 2: Version 1 + input in shared memory + unroll + TILE_WIDTH modified for each layer\n");
    // Calculate output dimensions
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float* d_in, * d_out, * d_weight;
    CHECK(cudaMalloc((void**)&d_in, n * in_channel * height_in * width_in * sizeof(float)));                      // input features map is in_channel
    CHECK(cudaMalloc((void**)&d_out, n * out_channel * height_out * width_out * sizeof(float)));                  // output feature map is out_channel
    CHECK(cudaMalloc((void**)&d_weight, out_channel * in_channel * kernel_height * kernel_height * sizeof(float)));  // in_channel * out_channel filter Maps of size height_kernel * height_kernel

    // Copy input and mask data to device
    CHECK(cudaMemcpy(d_in, in_data, n * in_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight, weight_data, out_channel * in_channel * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice));

    // Set the kernel dimensions and call the kernel
    int grid_z = (height_out + TILE_WIDTH - 1) / TILE_WIDTH * ((width_out + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridSize(n, out_channel, grid_z);

    // Launch the kernel
    GpuTimer timer;
    timer.Start();
    conv_forward_kernel << <gridSize, blockSize >> > (d_out, d_in, d_weight,
        out_channel, in_channel, height_in, width_in, kernel_height);

    timer.Stop();
    std::cout << "Kernel time: " << timer.Elapsed() << " ms\n";
    // Copy the output back to host
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    cudaDeviceSynchronize();

    // Copy output data back to host
    CHECK(cudaMemcpy(out_data, d_out, n * out_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_weight));
}