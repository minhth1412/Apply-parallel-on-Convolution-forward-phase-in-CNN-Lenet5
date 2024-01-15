# Compiler and Flags
NVCC := nvcc
CUDA_LIBRARIES := -lm -lcuda -lrt
CUDA_INCLUDE := -I./ -L/usr/local/cuda/lib64 -lcudart

# Directories
SRC_DIR := src
LAYER_DIR := $(SRC_DIR)/layer
LOSS_DIR := $(SRC_DIR)/loss
OPTIMIZER_DIR := $(SRC_DIR)/optimizer
CUSTOM_DIR := $(LAYER_DIR)/custom

main: main.o build_dnn.o $(SRC_DIR)/network.o $(SRC_DIR)/mnist.o $(LAYER_DIR)/*.o $(LOSS_DIR)/*.o $(OPTIMIZER_DIR)/*.o
	$(NVCC) -o $@ $(CUDA_LIBRARIES) $^ $(CUDA_INCLUDE)

train: train.o $(SRC_DIR)/network.o $(SRC_DIR)/mnist.o $(LAYER_DIR)/*.o $(LOSS_DIR)/*.o $(OPTIMIZER_DIR)/*.o
	$(NVCC) -o $@ $(CUDA_LIBRARIES) $^ $(CUDA_INCLUDE)

build_dnn.o: build_dnn.cc
	$(NVCC) --compile $< $(CUDA_INCLUDE)

$(SRC_DIR)/network.o: $(SRC_DIR)/network.cc
	$(NVCC) --compile $< -o $@ $(CUDA_INCLUDE)

$(SRC_DIR)/mnist.o: $(SRC_DIR)/mnist.cc
	$(NVCC) --compile $< -o $@ $(CUDA_INCLUDE)

$(LAYER_DIR)/%.o: $(LAYER_DIR)/%.cc
	$(NVCC) --compile $< -o $@ $(CUDA_INCLUDE)

$(LOSS_DIR)/%.o: $(LOSS_DIR)/%.cc
	$(NVCC) --compile $< -o $@ $(CUDA_INCLUDE)

$(OPTIMIZER_DIR)/%.o: $(OPTIMIZER_DIR)/%.cc
	$(NVCC) --compile $< -o $@ $(CUDA_INCLUDE)

custom: $(CUSTOM_DIR)/gpu_utils.o $(CUSTOM_DIR)/gpu_conv_forward_v0.o #$(CUSTOM_DIR)/gpu_conv_forward_v1.o $(CUSTOM_DIR)/gpu_conv_forward_v2.o

$(CUSTOM_DIR)/%.o: $(CUSTOM_DIR)/%.cu
	$(NVCC) --compile $< -o $@ $(CUDA_INCLUDE)

# Clean
clean:
	rm -f train main

clean_o:
	rm -f *.o $(SRC_DIR)/*.o $(LAYER_DIR)/*.o $(LOSS_DIR)/*.o $(OPTIMIZER_DIR)/*.o $(CUSTOM_DIR)/*.o

setup:
	make clean_o
	make clean
	make network.o
	make mnist.o
	make layer
	make loss
	make optimizer
