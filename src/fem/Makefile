CXX = clang++
CXXFLAGS = -std=c++17 -O3 -mp -acc -gpu=cc90 -gpu=managed -Minfo=accel -Minfo=mp
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