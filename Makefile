CC = gcc
CXX = g++
CCFLAGS = -Wall -Wextra -O3
CXXFLAGS = -Wall -Wextra -O3 -std=c++17 `pkg-config --cflags opencv4`
LDFLAGS = -O3 `pkg-config --libs opencv4`
TARGET = plague-simulator

SRC_CPP = $(wildcard *.cpp)
SRC_CU = $(wildcard *.cu)
SRC_C = $(wildcard *.c)
OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
OBJ_C = $(SRC_C:.c=.o)
OBJ = $(OBJ_CPP) $(OBJ_CU) $(OBJ_C) 

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@ 

%.o: %.cu
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
