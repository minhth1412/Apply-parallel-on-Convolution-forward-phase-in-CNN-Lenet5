#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "build_dnn.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/layer/custom/gpu_utils.h"

int main(int argc, char** argv)
{
	printDeviceInfo();
	int gpu_version = -1;
	if (argc > 1) {
		gpu_version = std::atoi(argv[1]);
	}
	MNIST dataset("./data/");
	dataset.read();
	int n_train = dataset.train_data.cols();
	int dim_in = dataset.train_data.rows();
	float accuracy = 0.0;
	std::cout << "mnist train number: " << n_train << std::endl;
	std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
	std::cout << "-----------------------------------------------\n";

	// Run dnn with convolutional layers running on CPU
	if (gpu_version == -1) {
		std::cout << "Network using host:\n";
		Network dnn = dnnNetwork();
		dnn.load_parameters("./model/trained_model_32_64.bin");
		dnn.forward(dataset.test_data);
		accuracy = compute_accuracy(dnn.output(), dataset.test_labels);
		std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";
	}
	// Run dnn with convolutional layers running on GPU
	else if (gpu_version == 0) {
		// Version 0: Basic GPU Convolution Kernel
		std::cout << "Version 0: Basic GPU Convolution Kernel\n";
		Network dnn = dnnNetwork_gpu();
		dnn.load_parameters("./model/trained_model_32_64.bin");
		dnn.forward(dataset.test_data);
		accuracy = compute_accuracy(dnn.output(), dataset.test_labels);
		std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";
	}
	else if (gpu_version == 1) {
		// Version 1: Basic GPU Convolution Kernel + Shared Memory
		std::cout << "Version 1: Basic GPU Convolution Kernel + Shared Memory\n";
		Network dnn = dnnNetwork_gpu();
		dnn.load_parameters("./model/trained_model_32_64.bin");
		dnn.forward(dataset.test_data);
		accuracy = compute_accuracy(dnn.output(), dataset.test_labels);
		std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";
	}
	else if (gpu_version == 2) {
		// Version 2: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters
		std::cout << "Version 2: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters\n";
		Network dnn = dnnNetwork_gpu();
		dnn.load_parameters("./model/trained_model_32_64.bin");
		dnn.forward(dataset.test_data);
		accuracy = compute_accuracy(dnn.output(), dataset.test_labels);
		std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";
	}
	else if (gpu_version == 3) {
		// Version 3: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters + Parallelize over images
		std::cout << "Version 3: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters + Parallelize over images\n";
		Network dnn = dnnNetwork_gpu();
		dnn.load_parameters("./model/trained_model_32_64.bin");
		dnn.forward(dataset.test_data);
		accuracy = compute_accuracy(dnn.output(), dataset.test_labels);
		std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";
	}

	return 0;
}