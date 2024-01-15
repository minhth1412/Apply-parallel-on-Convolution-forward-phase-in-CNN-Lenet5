network.o: src/network.cc
	nvcc --compile src/network.cc -o src/network.o -I src/ -I third_party/eigen

mnist.o: src/mnist.cc
	nvcc --compile src/mnist.cc -o src/mnist.o -I src/ -I third_party/eigen

layer: src/layer/conv.cc src/layer/conv_gpu.cc src/layer/ave_pooling.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc 
	nvcc --compile src/layer/ave_pooling.cc -o src/layer/ave_pooling.o -I./ -I third_party/eigen
	nvcc --compile src/layer/conv.cc -o src/layer/conv.o -I./ -I third_party/eigen
	nvcc --compile src/layer/conv_gpu.cc -o src/layer/conv_gpu.o -I./ -I third_party/eigen
	nvcc --compile src/layer/fully_connected.cc -o src/layer/fully_connected.o -I./ -I third_party/eigen
	nvcc --compile src/layer/max_pooling.cc -o src/layer/max_pooling.o -I./ -I third_party/eigen
	nvcc --compile src/layer/relu.cc -o src/layer/relu.o -I./ -I third_party/eigen
	nvcc --compile src/layer/sigmoid.cc -o src/layer/sigmoid.o -I./ -I third_party/eigen
	nvcc --compile src/layer/softmax.cc -o src/layer/softmax.o -I./ -I third_party/eigen

custom0:
	nvcc --compile src/layer/custom/gpu_conv_forward_v0.cu -o src/layer/custom/gpu_conv_forward_v0.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/custom/gpu_utils.cu -o src/layer/custom/gpu_utils.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart

custom1:
	nvcc --compile src/layer/custom/gpu_conv_forward_v1.cu -o src/layer/custom/gpu_conv_forward_v1.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/custom/gpu_utils.cu -o src/layer/custom/gpu_utils.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart

custom2:
	nvcc --compile src/layer/custom/gpu_conv_forward_v2.cu -o src/layer/custom/gpu_conv_forward_v2.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/custom/gpu_utils.cu -o src/layer/custom/gpu_utils.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart

custom3:
	nvcc --compile src/layer/custom/gpu_conv_forward_v3.cu -o src/layer/custom/gpu_conv_forward_v3.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart 
	nvcc --compile src/layer/custom/gpu_utils.cu -o src/layer/custom/gpu_utils.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart
		
loss: src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
	nvcc -arch=sm_75 --compile src/loss/cross_entropy_loss.cc -o src/loss/cross_entropy_loss.o -I./ -I third_party/eigen
	nvcc -arch=sm_75 --compile src/loss/mse_loss.cc -o src/loss/mse_loss.o -I./ -I third_party/eigen

optimizer: src/optimizer/sgd.cc
	nvcc -arch=sm_75 --compile src/optimizer/sgd.cc -o src/optimizer/sgd.o -I./ -I third_party/eigen

main: main.o custom0
	nvcc -o main -lm -lcuda -lrt main.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/*.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart 

main_custom1: main.o custom1
	nvcc -o main -lm -lcuda -lrt main.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/*.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart 

main_custom2: main.o custom2
	nvcc -o main -lm -lcuda -lrt main.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/*.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart 

main_custom3: main.o custom3
	nvcc -o main -lm -lcuda -lrt main.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/*.o -I./ -I third_party/eigen -L/usr/local/cuda/lib64 -lcudart 

main.o: main.cc
	nvcc --compile main.cc -I./ -I third_party/eigen

clean:
	rm -f main train 
	rm -f *.o src/*.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/custom/*.o

setup:
	make network.o
	make mnist.o
	make layer
	make loss
	make optimizer

run: main
	./main

run0: main
	./main 0

run1: main_custom1
	./main 1

run2: main_custom2
	./main 2

run3: main_custom3
	./main 3