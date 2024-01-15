#include "conv_gpu.h"
#include <math.h>
#include <iostream>
#include "custom/gpu_utils.h"
#include "custom/gpu_conv_forward.h"


void Conv_gpu::init() {
	height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
	width_out = (1 + (width_in - width_kernel + 2 * pad_w) / stride);
	dim_out = height_out * width_out * channel_out;

	weight.resize(channel_in * height_kernel * width_kernel, channel_out);
	bias.resize(channel_out);
	grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
	grad_bias.resize(channel_out);
	set_normal_random(weight.data(), weight.size(), 0, 0.01);
	set_normal_random(bias.data(), bias.size(), 0, 0.01);
	//std::cout << weight.colwise().sum() << std::endl;
	//std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv_gpu::im2col(const Vector& image, Matrix& data_col) {
	int hw_in = height_in * width_in;
	int hw_kernel = height_kernel * width_kernel;
	int hw_out = height_out * width_out;
	// im2col
	data_col.resize(hw_out, hw_kernel * channel_in);
	for (int c = 0; c < channel_in; c++) {
		Vector map = image.block(hw_in * c, 0, hw_in, 1);  // c-th channel map
		for (int i = 0; i < hw_out; i++) {
			int step_h = i / width_out;
			int step_w = i % width_out;
			int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
			for (int j = 0; j < hw_kernel; j++) {
				int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
				int cur_row = start_idx / width_in + j / width_kernel - pad_h;
				if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
					cur_row >= height_in) {
					data_col(i, c * hw_kernel + j) = 0;
				}
				else {
					//int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
					int pick_idx = cur_row * width_in + cur_col;
					data_col(i, c * hw_kernel + j) = map(pick_idx);  // pick which pixel
				}
			}
		}
	}
}

// In this project, the requirement is optimize the convolution layer using CUDA.
// And only need to code the inference phase (forward pass) of the convolution layer.
void Conv_gpu::forward(const Matrix& bottom) {
	// initialize
	int n_sample = bottom.cols();
	GpuTimer timer;
	top.resize(height_out * width_out * channel_out, n_sample);
	data_cols.resize(n_sample);
	// Base on channel_in in the Conv class, we can define which is layer 1 (1st convolution) and which is layer 3 (2nd one).
	std::cout << "Convolution layer " << (channel_in == 1) ? ("1") : ("3");
	timer.Start();
	std::cout << ", using device:\n";
	// Create constant variables for GPU kernel execution
	float* in_data = (float*)bottom.data();
	float* out_data = (float*)top.data();
	float* weight_data = (float*)weight.data();

	GPU_Conv_Forward gpu_conv_forward;

	gpu_conv_forward.execute(in_data, out_data, weight_data,
		n_sample, channel_in, channel_out, height_in, width_in, height_kernel);

	// Stop the timer
	timer.Stop();
	std::cout << "Total time running this layer: " timer.Elapsed() << " ms\n";
}

// col2im, used for grad_bottom
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
// image size: Vector (height_in * width_in * channel_in)
void Conv_gpu::col2im(const Matrix& data_col, Vector& image) {
	int hw_in = height_in * width_in;
	int hw_kernel = height_kernel * width_kernel;
	int hw_out = height_out * width_out;
	// col2im
	image.resize(hw_in * channel_in);
	image.setZero();
	for (int c = 0; c < channel_in; c++) {
		for (int i = 0; i < hw_out; i++) {
			int step_h = i / width_out;
			int step_w = i % width_out;
			int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
			for (int j = 0; j < hw_kernel; j++) {
				int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
				int cur_row = start_idx / width_in + j / width_kernel - pad_h;
				if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
					cur_row >= height_in) {
					continue;
				}
				else {
					//int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
					int pick_idx = cur_row * width_in + cur_col;
					image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j);  // pick which pixel
				}
			}
		}
	}
}


void Conv_gpu::backward(const Matrix& bottom, const Matrix& grad_top) {
	int n_sample = bottom.cols();
	grad_weight.setZero();
	grad_bias.setZero();
	grad_bottom.resize(height_in * width_in * channel_in, n_sample);
	grad_bottom.setZero();
	for (int i = 0; i < n_sample; i++) {
		// im2col of grad_top
		Matrix grad_top_i = grad_top.col(i);
		Matrix grad_top_i_col = Eigen::Map<Matrix>(grad_top_i.data(),
			height_out * width_out, channel_out);
		// d(L)/d(w) = \sum{ d(L)/d(z_i) * d(z_i)/d(w) }
		grad_weight += data_cols[i].transpose() * grad_top_i_col;
		// d(L)/d(b) = \sum{ d(L)/d(z_i) * d(z_i)/d(b) }
		grad_bias += grad_top_i_col.colwise().sum().transpose();
		// d(L)/d(x) = \sum{ d(L)/d(z_i) * d(z_i)/d(x) } = d(L)/d(z)_col * w'
		Matrix grad_bottom_i_col = grad_top_i_col * weight.transpose();
		// col2im of grad_bottom
		Vector grad_bottom_i;
		col2im(grad_bottom_i_col, grad_bottom_i);
		grad_bottom.col(i) = grad_bottom_i;
	}
}

void Conv_gpu::update(Optimizer& opt) {
	Vector::AlignedMapType weight_vec(weight.data(), weight.size());
	Vector::AlignedMapType bias_vec(bias.data(), bias.size());
	Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
	Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

	opt.update(weight_vec, grad_weight_vec);
	opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> Conv_gpu::get_parameters() const {
	std::vector<float> res(weight.size() + bias.size());
	// Copy the data of weights and bias to a long vector
	std::copy(weight.data(), weight.data() + weight.size(), res.begin());
	std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
	return res;
}

void Conv_gpu::set_parameters(const std::vector<float>& param) {
	if (static_cast<int>(param.size()) != weight.size() + bias.size())
		throw std::invalid_argument("Parameter size does not match");
	std::copy(param.begin(), param.begin() + weight.size(), weight.data());
	std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Conv_gpu::get_derivatives() const {
	std::vector<float> res(grad_weight.size() + grad_bias.size());
	// Copy the data of weights and bias to a long vector
	std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
	std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
		res.begin() + grad_weight.size());
	return res;
}