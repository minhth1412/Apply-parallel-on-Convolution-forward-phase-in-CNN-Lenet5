#include "gpu_conv_forward.h"
#include "gpu_utils.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH 16
#define UNROLL_FACTOR 4  // based on performance testing

// Create a constant memory for the filter
__constant__ float dc_filter[2048];

__global__ void conv_forward_kernel(float* d_out, const float* d_in, const float* kernel,
    const int out_channel, const int in_channel,
    const int height, const int width, const int kernel_size)
{
    const int height_out = height - kernel_size + 1;
    const int width_out = width - kernel_size + 1;

    int width_grid = (width_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int row_idx = (bz / width_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int col_idx = (bz % width_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix

    __shared__ float shared_input[TILE_WIDTH][TILE_WIDTH];

    float accumulator = 0.0f;

    for (int in_offset = 0; in_offset < in_channel; in_offset += UNROLL_FACTOR)
    {
        int in_row = row_idx;
        int in_col = col_idx + in_offset;

        // Unroll the input matrix
#pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; ++i)
        {
            if (in_row < height && in_col < width)
            {
                shared_input[threadIdx.y][threadIdx.x + i * TILE_WIDTH] = d_in[bx * (in_channel * height * width) + in_col * height + in_row];
            }
            else
            {
                shared_input[threadIdx.y][threadIdx.x + i * TILE_WIDTH] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        // Perform tiled convolution
        for (int i = 0; i < kernel_size; ++i)
        {
            for (int j = 0; j < kernel_size; ++j)
            {
                accumulator += shared_input[ty + i][tx + j] * dc_filter[by * (in_channel * kernel_size * kernel_size) + (i * kernel_size) + j];
            }
        }

    }

    int out_area = height_out * width_out;
    if (row_idx < height_out && col_idx < width_out)
    {
        d_out[bx * (out_channel * out_area) + by * out_area + row_idx * width_out + col_idx] = accumulator;
    }
}

void GPU_Conv_Forward::execute(const float* in_data, float* out_data, const float* weight_data,
    const int n, const int in_channel, const int out_channel,
    const int height_in, const int width_in, const int kernel_height)
{
    // Optimize version 3: Version 2 + tiled shared memory convolution
    printf("\nOptimization version 3: Version 2 + tiled shared memory convolution\n");

    // Calculate output dimensions
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float* d_in, * d_out;
    CHECK(cudaMalloc((void**)&d_in, n * in_channel * height_in * width_in * sizeof(float)));                      // input features map is in_channel
    CHECK(cudaMalloc((void**)&d_out, n * out_channel * height_out * width_out * sizeof(float)));                  // output feature map is out_channel

    // Copy input and mask data to device
    CHECK(cudaMemcpy(d_in, in_data, n * in_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(dc_filter, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float)));

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

    // Check for errors in kernel launch if any
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
}