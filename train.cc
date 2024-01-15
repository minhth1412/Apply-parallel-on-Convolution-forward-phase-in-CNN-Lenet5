/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>

#include "build_dnn.h"
#include "src/layer.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

int main(int argc, char* argv[]) {
	// Solved, using visual studio to run this program, the argv[0] is the path of the program, that 
	// check the path of the program if it is correct.
	std::cout << "current path: " << argv[0] << '\n';
	std::cout << "In case you are running this project without Visual Studio, you might need above path	for hard-core set up the filepath in this file.\n";
	std::cout << "Input: type the number of epoch that you want for training phase: n_epoch = ";
	int n_epoch;
	std::cin >> n_epoch;
	std::cout << "Input: type the batch size that you want for training phase: batch_size = ";
	int batch_size;
	std::cin >> batch_size;
	std::string path = "../../../model/trained_model_" + std::to_string(n_epoch) + "_" + std::to_string(batch_size) + ".bin";
	std::string log_path = "../../../model/logging/log_" + std::to_string(n_epoch) + "_" + std::to_string(batch_size) + ".txt";
	// Create a log file
	std::ofstream logFile(log_path);
	// Redirect std::cout to the log file, but it will not display on the console, so I need to 
	// use printf to display on the console, it will be a little bit ugly but it needs for eye debugging
	std::streambuf* coutBuffer = std::cout.rdbuf();
	std::cout.rdbuf(logFile.rdbuf());

	// data
	printf("Time for training data is quite long, I already save my trained model in the path %s\n", path.c_str());
	printf("If you want more details about training stage, the logging file is yours to check.\nIt is located at %s\n", log_path.c_str());

	// use this path "../data/" or "./data" when run in google colab, make sure to check the path of the data
	//MNIST dataset("./data/");
	// use this path "../../../data/" when run in visual studio
	MNIST dataset("../../../data/");
	dataset.read();
	int n_train = dataset.train_data.cols();
	int dim_in = dataset.train_data.rows();
	std::cout << "mnist train number: " << n_train << std::endl;
	printf("mnist train number: %d\n", n_train);
	std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
	printf("mnist test number: %d\n", dataset.test_labels.cols());
	// dnn, false means not using gpu for creating dnn
	Network dnn = dnnNetwork(false);

	// train
	SGD opt(0.01, 5e-4, 0.9, true);		// learning rate = 0.01, weight_decay = 5e-4, momentum = 0.9, nesterov = true
	// SGD opt(0.001);
	
	// Bigger n_epoch will take more time to train but it will create a model having higher training accurancy.
	// Also test batch size here for best model with highest training accurancy, choose exponential of 2 {32, 64, 128}.
	// In this project I choose 64 for batch size, and 32 for n_epoch, it takes about 800 minutes to train.
	// If I have more time, I will find the way to train model with python using GPU in google colab, it will be much faster.
	
	for (int epoch = 0; epoch < n_epoch; epoch++) {
		shuffle_data(dataset.train_data, dataset.train_labels);
		for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
			int ith_batch = start_idx / batch_size;
			Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
				std::min(batch_size, n_train - start_idx));
			Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
				std::min(batch_size, n_train - start_idx));
			Matrix target_batch = one_hot_encode(label_batch, 10);
			if (false && ith_batch % 10 == 1) {
				std::cout << ith_batch << "-th grad: " << std::endl;
				printf("%d-th grad: \n", ith_batch);
				dnn.check_gradient(x_batch, target_batch, 10);
			}
			dnn.forward(x_batch);
			dnn.backward(x_batch, target_batch);
			// display
			if (ith_batch % 50 == 0) {
				std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss()
					<< std::endl;
				printf("%d-th batch, loss: %f\n", ith_batch, dnn.get_loss());
			}
			// optimize
			dnn.update(opt);
		}
		// test
		dnn.forward(dataset.test_data);
		float acc = compute_accuracy(dnn.output(), dataset.test_labels);
		std::cout << std::endl;
		if (epoch == 0) {
			std::cout << "1-st epoch, test acc: " << acc << std::endl;
			printf("1-st epoch, test acc: %f\n", acc);
		}
		else if (epoch == 1) {
			std::cout << "2-nd epoch, test acc: " << acc << std::endl;
			printf("2-nd epoch, test acc: %f\n", acc);
		}
		else if (epoch == 2) {
			std::cout << "3-rd epoch, test acc: " << acc << std::endl;
			printf("3-rd epoch, test acc: %f\n", acc);
		}
		else {
			std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
			printf("%d-th epoch, test acc: %f\n", epoch + 1, acc);
		}
		std::cout << std::endl;
	}
	// Restore std::cout to the original buffer
	std::cout.rdbuf(coutBuffer);
	// Close the log file
	logFile.close();

	// Save the trained model into a file, with the name "trained_model_<n_epoch>_<batch_size>.bin" to choose the best model that has the highest training accuracy
	dnn.save_parameters(path);
	return 0;
}

