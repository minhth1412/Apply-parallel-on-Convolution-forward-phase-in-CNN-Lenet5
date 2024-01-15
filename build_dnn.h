#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <cstdlib>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"

Network dnnNetwork(int use_gpu = -1)
{
	// define dnn network
	// Setup network structure for training, following the modified LeNet-5 architecture that is given in the final project description
	 /*
	 model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))		// other parameters are default: stride = 1, no padding
	 model.add(MaxPooling2D(pool_size=(2, 2)))													// Default stride is pool_size, which is (2, 2) here
	 model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))								// other parameters are default: stride = 1, no padding, input_shape is inferred from previous layer
	 model.add(MaxPooling2D(pool_size=(2, 2)))
	 model.add(Flatten())
	 model.add(Dense(120, activation='relu'))
	 model.add(Dense(84, activation='relu'))
	 model.add(Dense(10, activation='softmax'))
	 */
	Network dnn;
	Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5, use_gpu);				// 1 input channel, 28x28 input image, 6 output channels, 5x5 kernel size
	Layer* conv2 = new Conv(6, 12, 12, 16, 5, 5, use_gpu);				// 6 input channels, 12x12 input image, 16 output channels, 5x5 kernel size
	Layer* pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);					// 6 input channels, 24x24 input image, 2x2 kernel size, stride = 2
	Layer* pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);					// 16 input channels, 8x8 input image, 2x2 kernel size, stride = 2
	// In this place, it is not necessary to create a flatten layer, because the fully connected layer can automatically flatten the input
	Layer* fc1 = new FullyConnected(pool2->output_dim(), 120);			// pool2->output_dim() = 1024, it returns the flattened dimension of the output of the previous layer (pool2)
	Layer* fc2 = new FullyConnected(120, 84);
	Layer* fc3 = new FullyConnected(84, 10);
	Layer* relu1 = new ReLU;
	Layer* relu2 = new ReLU;
	Layer* relu3 = new ReLU;
	Layer* relu4 = new ReLU;
	Layer* softmax = new Softmax;
	// Add layers to the network, following the modified LeNet-5 architecture that is given in the final project description
	dnn.add_layer(conv1);					// After running, the output_dim is 6x24x24
	dnn.add_layer(relu1);
	dnn.add_layer(pool1);   				// After running, the output_dim is 6x12x12
	dnn.add_layer(conv2);					// After running, the output_dim is 16x8x8
	dnn.add_layer(relu2);
	dnn.add_layer(pool2);   				// After running, the output_dim is 16x4x4
	dnn.add_layer(fc1);
	dnn.add_layer(relu3);
	dnn.add_layer(fc2);
	dnn.add_layer(relu4);
	dnn.add_layer(fc3);
	dnn.add_layer(softmax);

	// loss
	Loss* loss = new CrossEntropy;
	dnn.add_loss(loss);
	return dnn;
}