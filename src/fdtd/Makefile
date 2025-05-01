CXX = g++
CXXFLAGS = -std=c++17 -O2
TARGET = fdtd_sim
OBJS = main.o fdtd_simulation.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp fdtd_simulation.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp

fdtd_simulation.o: fdtd_simulation.cpp fdtd_simulation.hpp
	$(CXX) $(CXXFLAGS) -c fdtd_simulation.cpp

clean:
	rm -f $(TARGET) *.o observation.csv
