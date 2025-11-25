CXX = mpic++
CXXFLAGS = -std=c++17 -O3 -mp -acc -gpu=cc80 -gpu=mem:managed -Minfo=accel -Minfo=mp -I/home/kusumoto/include/Eigen
LDFLAGS =

TARGET ?= simulation.o
SRCS = main.cpp fem_simulation.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)