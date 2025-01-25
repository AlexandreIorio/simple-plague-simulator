CC = gcc
CXX = g++
CUDACC = nvcc
CCFLAGS = -Wall -Wextra -O3
CXXFLAGS = -Wall -Wextra -O3 -std=c++17 `pkg-config --cflags opencv4`
CUDAFLAGS = -lcudart
LDFLAGS = -O3
TARGET_BASE = plague-simulator-base
TARGET_CUDA = plague-simulator-cuda

SRC_CPP = $(wildcard *.cpp)
SRC_CU = $(wildcard *.cu)
SRC_C = $(wildcard *.c)

COMMON_OBJS = world_priv.o timeline.o main.o

BASE_OBJS = $(COMMON_OBJS) world.o
CUDA_OBJS = $(COMMON_OBJS) world.co

all: $(TARGET_BASE) $(TARGET_CUDA)

$(TARGET_BASE): $(BASE_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ 

$(TARGET_CUDA): $(CUDA_OBJS)
	$(CUDACC) $(LDFLAGS) -o $@ $^ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@ 

%.co: %.cu
	$(CUDACC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(BASE_OBJS) $(CUDA_OBJS) 
