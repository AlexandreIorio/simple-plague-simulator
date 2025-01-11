CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
TARGET = PlagueSimulator

SRC_CPP = $(wildcard *.cpp)
SRC_C = $(wildcard *.c)
OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_C = $(SRC_C:.c=.o)
OBJ = $(OBJ_CPP) $(OBJ_C)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@  # Utiliser g++ pour les fichiers C

clean:
	rm -f $(OBJ) $(TARGET)
