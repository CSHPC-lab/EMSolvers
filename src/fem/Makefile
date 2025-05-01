CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp
LDFLAGS = 

TARGET = fem_simulation
SRCS = main.cpp fem_simulation.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean 