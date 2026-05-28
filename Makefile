CXX      = clang++
CXXFLAGS = -std=c++20 -O3 -Wall -fobjc-arc
FRAMEWORKS = \
	-framework Metal \
	-framework Foundation \
	-framework MetalPerformanceShaders \
	-framework MetalPerformanceShadersGraph

TARGET  = GPUDBMPSBenchmark
SRCDIR  = src
BUILDDIR = build

SRCS = $(wildcard $(SRCDIR)/*.mm)
OBJS = $(patsubst $(SRCDIR)/%.mm, $(BUILDDIR)/%.o, $(SRCS))

.PHONY: all run clean

all: $(BUILDDIR) $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.mm
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -c -o $@ $<

run: all
	./$(TARGET)

clean:
	rm -rf $(BUILDDIR) $(TARGET)
