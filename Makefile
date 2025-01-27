CC = gcc
CXX = g++
CUDACC = nvcc
CCFLAGS = -Wall -Wextra -O3
CXXFLAGS = -Wall -Wextra -O3 -std=c++17
CUDAFLAGS = --compiler-options '$(CCFLAGS)'
LDFLAGS = -O3

TARGET_BASE = plague-simulator-base
TARGET_OMP = plague-simulator-omp
TARGET_CUDA = plague-simulator-cuda
TARGET_CUDA_OMP = plague-simulator-cuda-omp

SRC_CPP = $(wildcard *.cpp)
SRC_CU = $(wildcard *.cu)
SRC_C = $(wildcard *.c)

OBJS = world.o world_priv.o world_common.o timeline.o main.o

all: $(TARGET_BASE) $(TARGET_CUDA) $(TARGET_OMP) $(TARGET_CUDA_OMP)

$(TARGET_BASE): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ 

$(TARGET_OMP): $(OBJS)
	$(CXX) $(LDFLAGS) -fopenmp -o $@ $^ 

$(TARGET_CUDA): 
	./cuda_build.sh $(TARGET_CUDA)

$(TARGET_CUDA_OMP): 
	./cuda_build.sh $(TARGET_CUDA) -fopenmp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -f $(OBJS) $(TARGET_BASE) $(TARGET_CUDA) $(TARGET_CUDA_OMP) $(TARGET_OMP)
