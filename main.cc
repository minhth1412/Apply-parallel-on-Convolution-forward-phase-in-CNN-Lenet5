#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "build_dnn.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/layer/custom/gpu_utils.h"

int main()
{
	printDeviceInfo();

	MNIST dataset("./data/");
	dataset.read();
	int n_train = dataset.train_data.cols();
	int dim_in = dataset.train_data.rows();
	float accuracy = 0.0;
	std::cout << "mnist train number: " << n_train << std::endl;
	std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
	std::cout << "-----------------------------------------------\n";

	// Run dnn with convolutional layers running on CPU
	std::cout << "Network using host:\n";
	Network dnn = dnnNetwork();
	dnn.load_parameters("./model/trained_model_32_64.bin");
	dnn.forward(dataset.test_data);
	accuracy = compute_accuracy(dnn.output(), dataset.test_labels);
	std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";

	// Run dnn with convolutional layers running on GPU

	// Version 0: Basic GPU Convolution Kernel
	std::cout << "Network using device on convolutional layers:\n";
	std::cout << "Version 0: Basic GPU Convolution Kernel\n";
	Network dnn2 = dnnNetwork(0);
	dnn2.load_parameters("./model/trained_model_32_64.bin");
	dnn2.forward(dataset.test_data);
	accuracy = compute_accuracy(dnn2.output(), dataset.test_labels);
	std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";

	// Version 1: Basic GPU Convolution Kernel + Shared Memory
#if __CUDA_ARCH__ >= 300
	std::cout << "Network using device on convolutional layers:\n";
	std::cout << "Version 1: Basic GPU Convolution Kernel + Shared Memory\n";
	Network dnn3 = dnnNetwork(1);
	dnn3.load_parameters("./model/trained_model_32_64.bin");
	dnn3.forward(dataset.test_data);
	accuracy = compute_accuracy(dnn3.output(), dataset.test_labels);
	std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";
#endif

	// Version 2: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters
#if __CUDA_ARCH__ >= 300
	std::cout << "Network using device on convolutional layers:\n";
	std::cout << "Version 2: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters\n";
	Network dnn4 = dnnNetwork(true, true, true);
	dnn4.load_parameters("./model/trained_model_32_64.bin");
	dnn4.forward(dataset.test_data);
	accuracy = compute_accuracy(dnn4.output(), dataset.test_labels);
	std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";
#endif

	// Version 3: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters + Parallelize over images
#if __CUDA_ARCH__ >= 300
	std::cout << "Network using device on convolutional layers:\n";
	std::cout << "Version 3: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters + Parallelize over images\n";
	Network dnn5 = dnnNetwork(true, true, true, true);
	dnn5.load_parameters("./model/trained_model_32_64.bin");
	dnn5.forward(dataset.test_data);
	accuracy = compute_accuracy(dnn5.output(), dataset.test_labels);
	std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";
#endif

	// Version 4: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters + Parallelize over images + Parallelize over channels
#if __CUDA_ARCH__ >= 300
	std::cout << "Network using device on convolutional layers:\n";
	std::cout << "Version 4: Basic GPU Convolution Kernel + Shared Memory + Parallelize over filters + Parallelize over images + Parallelize over channels\n";
	Network dnn6 = dnnNetwork(true, true, true, true, true);
	dnn6.load_parameters("./model/trained_model_32_64.bin");
	dnn6.forward(dataset.test_data);
	accuracy = compute_accuracy(dnn6.output(), dataset.test_labels);
	std::cout << "Test accuracy: " << accuracy << "\n---------------------------------------------- - \n";
#endif


	return 0;
}