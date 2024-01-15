# Source
Copyright of **mini-dnn-cpp**, which is a C++ demo of deep neural networks. It is implemented purely in C++, whose only dependency, Eigen, is header-only. See more details on LICENSE file.

# Motivation
This is a final project of my course: Introduce to Parallel Programming at Ho Chi Minh city University of Science.

## Usage
Create a notebook file on [google colab](https://colab.research.google.com/), then type these in shell:

```shell
from IPython.display import clear_output
!git clone https://github.com/minhth1412/Apply-parallel-on-Convolution-forward-phase-in-CNN-Lenet5
!cd mini-dnn-cpp-master && make clean
!cd mini-dnn-cpp-master && make setup

# Use clear_output to clear the cell's output
clear_output(wait=True)

!cd mini-dnn-cpp-master && make run
```

```shell
!cd mini-dnn-cpp-master && make clean
!cd mini-dnn-cpp-master && make setup

# Use clear_output to clear the cell's output
clear_output(wait=True)

!cd mini-dnn-cpp-master && make run0
```
