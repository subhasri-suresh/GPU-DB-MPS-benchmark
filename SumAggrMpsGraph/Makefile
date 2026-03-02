CC      = xcrun clang
CFLAGS  = -fobjc-arc \
           -framework Foundation \
           -framework Metal \
           -framework MetalPerformanceShaders \
           -framework MetalPerformanceShadersGraph \
           -lc++

TARGET  = MetalVectorSum
SRC     = main.mm

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
